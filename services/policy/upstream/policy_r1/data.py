# 역할: 정책 문서와 질문·정답 JSONL을 읽고 데이터 누락·중복·평가셋 겹침을 검사합니다.
import json
import hashlib
import warnings
import unicodedata
from pathlib import Path


def question_key(question):
    text = unicodedata.normalize("NFKC", question).casefold()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def dataset_identity(config, corpus):
    """Detect changed QA order/content before restoring the saved shuffle indices."""
    hashes = {}
    paths = {"train": config.train_path, "val": config.val_path, "test": config.test_path}
    if config.retriever != "sqlite_bm25":
        paths["corpus"] = config.corpus_path
    for name, path in paths.items():
        if not path:
            continue
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    if config.retriever == "sqlite_bm25":
        hashes["index_metadata"] = corpus.metadata
    return hashes


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: 잘못된 JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: JSON object가 필요합니다.")
            rows.append(row)
    if not rows:
        raise ValueError(f"비어 있는 데이터: {path}")
    ids = [r.get("id") for r in rows]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"각 행에는 고유한 문자열 id가 필요합니다: {path}")
    return rows


def load_corpus(path):
    rows = read_jsonl(path)
    for row in rows:
        if not isinstance(row.get("contents"), str) or not row["contents"].strip():
            raise ValueError(f"문서 {row['id']}: contents 문자열이 필요합니다.")
    return rows


def load_qa(path):
    rows = read_jsonl(path)
    for row in rows:
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"질문 {row['id']}: question 문자열이 필요합니다.")
        answers = row.get("answers")
        if not isinstance(answers, list) or not answers or any(
            not isinstance(a, str) or not a.strip() for a in answers
        ):
            raise ValueError(f"질문 {row['id']}: 비어 있지 않은 정답 문자열 목록이 필요합니다.")
    return rows


def load_datasets(config):
    if config.retriever == "sqlite_bm25":
        from .sqlite_search import SQLiteRetriever
        corpus = SQLiteRetriever(config.corpus_path)
        corpus_demo = corpus.metadata.get("has_demo", False)
        document_ids = corpus
    else:
        corpus = load_corpus(config.corpus_path)
        corpus_demo = any(r.get("is_demo") for r in corpus)
        document_ids = {r["id"] for r in corpus}
    train, val = load_qa(config.train_path), load_qa(config.val_path)
    test = load_qa(config.test_path) if config.test_path else []
    if corpus_demo or any(r.get("is_demo") for rows in (train, val, test) for r in rows):
        if not config.allow_demo:
            raise ValueError("가상 예제 데이터가 포함되어 있습니다. 실제 데이터로 교체하세요.")
        warnings.warn("가상 정책 데이터: 파이프라인 확인용이며 정책 사실·성능 평가용이 아닙니다.")
    seen_ids, seen_questions = set(), set()
    for name, rows in (("train", train), ("val", val), ("test", test)):
        ids = {r["id"] for r in rows}
        questions = {question_key(r["question"]) for r in rows}
        if ids & seen_ids or questions & seen_questions:
            raise ValueError(f"{name}: split 간 id 또는 질문이 겹칩니다.")
        seen_ids.update(ids)
        seen_questions.update(questions)
    for row in train + val + test:
        evidence = row.get("evidence_ids", [])
        if not isinstance(evidence, list) or any(d not in document_ids for d in evidence):
            raise ValueError(f"질문 {row['id']}: corpus에 없는 evidence_ids입니다.")
    return corpus, train, val
