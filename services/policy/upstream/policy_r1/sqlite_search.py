"""Disk-backed Wikipedia FTS5/BM25 index; corpus is never loaded into a Python list."""
import gzip
import codecs
import io
import json
from contextlib import ExitStack, closing
from pathlib import Path
import re
import sqlite3
import tempfile
import tarfile
from collections import OrderedDict
from time import perf_counter


STOP_WORDS = frozenset("a an the and or of to in on at for from by with is are was were be been being "
                       "do does did has have had it its this that these those what which who whom "
                       "when where how".split())


def iter_corpus_lines(source):
    """Read JSONL, gzip JSONL, or tar(.gz) by content, without extracting files.

    The official wiki-18.jsonl.gz contains a GNU tar header followed by
    data00/.../wiki_dump.jsonl, despite its filename. Its binary size field
    must never be decoded as UTF-8 text.
    """
    with ExitStack() as stack:
        raw = stack.enter_context(Path(source).open("rb"))
        compressed = raw.peek(2)[:2] == b"\x1f\x8b"
        binary = stack.enter_context(gzip.GzipFile(fileobj=raw)) if compressed else raw
        buffered = stack.enter_context(io.BufferedReader(binary))
        header = buffered.peek(512)[:512]
        try:
            is_tar = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape") is not None
        except tarfile.HeaderError:
            is_tar = False
        if not is_tar:
            print(f"Corpus format: {'gzip JSONL' if compressed else 'JSONL'}", flush=True)
            text = stack.enter_context(io.TextIOWrapper(buffered, encoding="utf-8-sig"))
            yield from text
            return
        print(f"Corpus format: {'tar.gz' if compressed else 'tar'} (streaming JSONL members)", flush=True)
        archive = stack.enter_context(tarfile.open(fileobj=buffered, mode="r|"))
        found = False
        for member in archive:
            # No extract()/extractall(): archive paths and links cannot write to disk.
            if not member.isfile() or not member.name.lower().endswith(".jsonl"):
                continue
            found = True
            print(f"Corpus member: {member.name} ({member.size:,} bytes)", flush=True)
            with archive.extractfile(member) as payload:
                # TextIOWrapper asks seekable(), unsupported by tar's stream
                # adapter on Python 3.10/3.11. This decoder only reads forward.
                yield from codecs.getreader("utf-8-sig")(payload)
        if not found:
            raise ValueError(f"{source}: tar archive contains no regular .jsonl file")
        # Consume the gzip trailer too: a truncated/corrupt download must not
        # publish a seemingly complete index, even after tar's end marker.
        while buffered.read(1024 * 1024):
            pass


def build_sqlite_index(corpus_path, index_path, batch_size=10000):
    source, destination = Path(corpus_path), Path(index_path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    stat = source.stat()
    signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if destination.exists():
        existing = SQLiteRetriever(destination)
        try:
            if existing.metadata["source_signature"] != signature:
                raise FileExistsError(f"Index source differs: {destination}; use a new index path")
            return existing.metadata
        finally:
            existing.close()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".wiki-", suffix=".sqlite", dir=destination.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    connection = sqlite3.connect(temporary)
    count, has_demo = 0, False
    try:
        connection.executescript("""
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=FILE;
            PRAGMA cache_size=-65536;
            CREATE TABLE documents (rowid INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, contents TEXT NOT NULL);
            CREATE VIRTUAL TABLE passages USING fts5(contents, content='documents', content_rowid='rowid',
                                                     tokenize='porter unicode61');
            CREATE TABLE metadata (value TEXT NOT NULL);
        """)
        batch = []
        with closing(iter_corpus_lines(source)) as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                contents = row.get("contents")
                if contents is None and isinstance(row.get("text"), str):
                    contents = (str(row.get("title") or "") + "\n" + row["text"]).strip()
                if row.get("id") is None or not isinstance(contents, str) or not contents.strip():
                    raise ValueError(f"{source}:{line_number}: id and nonempty contents (or text) are required")
                has_demo = has_demo or bool(row.get("is_demo"))
                count += 1
                batch.append((count, str(row["id"]), contents))
                if len(batch) >= batch_size:
                    _insert(connection, batch)
                    batch.clear()
                    print(f"Indexed {count:,} passages", flush=True)
        if batch:
            _insert(connection, batch)
        if count == 0:
            raise ValueError("Corpus is empty")
        metadata = {"version": 1, "complete": True, "documents": count, "has_demo": has_demo,
                    "source": str(source.resolve()), "source_signature": signature, "retriever": "sqlite_fts5_bm25"}
        connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
        connection.commit()
        connection.close()
        temporary.rename(destination)
        return metadata
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)


def _insert(connection, rows):
    with connection:
        connection.executemany("INSERT INTO documents VALUES (?, ?, ?)", rows)
        connection.executemany("INSERT INTO passages(rowid, contents) VALUES (?, ?)",
                               [(row[0], row[2]) for row in rows])


class SQLiteRetriever:
    def __init__(self, path, cache_mib=256, mmap_mib=1024, result_cache_size=128):
        if min(cache_mib, mmap_mib, result_cache_size) < 0:
            raise ValueError("Search cache limits must be nonnegative")
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"검색 인덱스가 없습니다: {path}. scripts/prepare_search_r1.py --stage index를 실행하세요.")
        self.connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            self.metadata = json.loads(self.connection.execute("SELECT value FROM metadata").fetchone()[0])
            if not self.metadata.get("complete") or self.metadata.get("version") != 1:
                raise ValueError("Incomplete or unsupported search index")
            self.connection.execute(f"PRAGMA cache_size=-{int(cache_mib * 1024)}")
            self.connection.execute(f"PRAGMA mmap_size={int(mmap_mib * 1024 * 1024)}")
            # Explicit BM25 + LIMIT lets SQLite keep a bounded top-k sorter.
            # Preserve a custom rank mapping if an external index has one.
            rank = self.connection.execute("SELECT v FROM passages_config WHERE k='rank'").fetchone()
            self._scorer = "bm25(passages)" if rank is None or rank[0] == "bm25()" else "rank"
            self._cache = OrderedDict()
            self._cache_limit = result_cache_size
            self.search_seconds = 0.0
            self.search_calls = 0
            self.cache_hits = 0
        except Exception:
            self.close()
            raise

    def __len__(self):
        return self.metadata["documents"]

    def __contains__(self, document_id):
        return self.connection.execute("SELECT 1 FROM documents WHERE id=?", (document_id,)).fetchone() is not None

    def search(self, query, top_k=3):
        started = perf_counter()
        self.search_calls += 1
        try:
            return self._search(query, top_k)
        finally:
            self.search_seconds += perf_counter() - started

    def _search(self, query, top_k):
        # Quote lexical terms; model output must never become SQL/FTS syntax.
        tokens = list(dict.fromkeys(re.findall(r"[^\W_]+", query.casefold())))[:64]
        terms = [term for term in tokens if term not in STOP_WORDS] or tokens
        if not terms or top_k < 1:
            return []
        expression = " OR ".join('"' + term + '"' for term in terms)
        key = (expression, top_k)
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return [dict(hit) for hit in self._cache[key]]
        ranked = self.connection.execute(
            f"SELECT rowid, {self._scorer} AS score FROM passages "
            "WHERE passages MATCH ? ORDER BY score LIMIT ?", (expression, top_k)).fetchall()
        # Fetch document bodies only for the final top-k, preserving rank order.
        hits = []
        for rowid, score in ranked:
            document = self.connection.execute(
                "SELECT id, contents FROM documents WHERE rowid=?", (rowid,)).fetchone()
            if document is None:
                raise ValueError(f"FTS index references missing document {rowid}")
            hits.append({"id": document[0], "contents": document[1], "score": -score})
        if self._cache_limit:
            self._cache[key] = tuple(dict(hit) for hit in hits)
            if len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
        return hits

    def close(self):
        self.connection.close()
