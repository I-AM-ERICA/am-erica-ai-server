"""Public AARSynth snapshot and leakage-free deterministic data partitions."""

import json
import random
import re
from dataclasses import asdict
from pathlib import Path

from .io import digest, write_json
from .types import Document, Review

DATASET_REPO = "recmeapp/AARSynth"
DATASET_FILE = "replies/AllRepliesWithRRGen.txt"


def normalize(text):
    return " ".join(text.casefold().split())


def clean_text(text):
    # Upstream anonymizes names and URLs. Do not fabricate replacements.
    return re.sub(r"\s+", " ", text.replace("<salutation>", "")).strip()


def parse_aarsynth(line):
    parts = line.rstrip("\r\n").split("***")
    if len(parts) != 7:
        return None
    app, rating, _date, _title, review, reply, _reply_date = parts
    review, reply = clean_text(review), clean_text(reply)
    if not app or rating not in {"1", "2", "3", "4", "5"}:
        return None
    if len(review.split()) < 4 or len(reply.split()) < 4 or reply.lower() == "none":
        return None
    return Review(digest([app, normalize(review)])[:24], review, app, reply)


def download_reviews(config):
    """Stream a bounded prefix; cache only eligible reviews, not the 236 MB corpus.

    This is a small reproducible demo sample, not a uniform sample of the full dataset.
    No synthetic/offline fallback is substituted if the download fails.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    spec = {
        "repository": DATASET_REPO,
        "revision": config.dataset_revision,
        "file": DATASET_FILE,
        "app_id": config.app_id,
        "download_limit": config.download_limit,
        "max_scan_lines": config.max_scan_lines,
        "parser_version": 1,
    }
    path = Path(config.data_dir) / f"aarsynth-{digest(spec)[:16]}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return [Review(**row) for row in cached["reviews"]], cached["provenance"]
    url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{config.dataset_revision}/{DATASET_FILE}"
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])),
    )
    rows, seen, scanned = [], set(), 0
    print(f"Downloading AARSynth ({config.app_id}, up to {config.download_limit} reviews)...", flush=True)
    with session, session.get(url, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        for scanned, raw in enumerate(response.iter_lines(), 1):
            row = parse_aarsynth(raw.decode("utf-8"))
            if row and row.app_id == config.app_id and row.id not in seen:
                rows.append(row)
                seen.add(row.id)
            if len(rows) >= config.download_limit or scanned >= config.max_scan_lines:
                break
    required = config.train_size + config.validation_size + config.test_size + config.knowledge_size
    if len(rows) < required:
        raise ValueError(
            f"Only {len(rows)} eligible reviews; need {required}. "
            "Increase download_limit/max_scan_lines or reduce split sizes."
        )
    provenance = {
        **spec,
        "url": url,
        "scanned_lines": scanned,
        "eligible_rows": len(rows),
        "source": "https://huggingface.co/datasets/recmeapp/AARSynth",
        "reference_type": "Historical developer replies; not independently verified expert labels",
        "sampling": "bounded source prefix, then seeded shuffle",
    }
    write_json(path, {"provenance": provenance, "reviews": [asdict(r) for r in rows]})
    return rows, provenance


def load_custom_reviews(path):
    """JSONL with id, review, app_id, and optional expert_response."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(Review(**json.loads(line)))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid custom JSONL line {i}: {exc}") from exc
    return rows


def split_reviews(rows, config):
    # Deduplicate globally by review text, not just IDs, before any split.
    unique, seen_text, seen_ids = [], set(), set()
    for row in rows:
        key = normalize(row.review)
        if key in seen_text or row.id in seen_ids:
            continue
        seen_text.add(key)
        seen_ids.add(row.id)
        unique.append(row)
    random.Random(config.seed).shuffle(unique)
    sizes = {
        "knowledge": config.knowledge_size,
        "train": config.train_size,
        "validation": config.validation_size,
        "test": config.test_size,
    }
    if len(unique) < sum(sizes.values()):
        raise ValueError(f"Need {sum(sizes.values())} unique reviews, got {len(unique)}")
    splits, offset = {}, 0
    for name, size in sizes.items():
        splits[name] = unique[offset : offset + size]
        offset += size
    return splits


def knowledge_documents(rows):
    return [
        Document(
            id=f"history-{r.id}",
            app_id=r.app_id,
            source=f"AARSynth:{r.id}",
            text=f"Historical support example (may be outdated, not a current policy).\n"
            f"App: {r.app_id}\nReview: {r.review}\nDeveloper reply: {r.expert_response}",
        )
        for r in rows
        if r.expert_response and r.expert_response.strip()
    ]


def load_documents(directory, app_id):
    """Load user-supplied TXT/MD/PDF documents as current store/product documentation.

    Files directly in `directory` belong to `app_id`. Files under a subfolder belong to the
    store/app named by that first-level folder, e.g. documents/store-001/menu.md -> "store-001",
    so each review only retrieves its own store's documents.
    """
    root = Path(directory)
    documents = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            text = path.read_text(encoding="utf-8")
        if text.strip():
            parts = path.relative_to(root).parts
            owner = parts[0] if len(parts) > 1 else app_id
            documents.append(Document(digest([str(path), text])[:24], text, owner, str(path)))
    if not documents:
        raise ValueError(f"No readable TXT/MD/PDF documents found in {directory}")
    return documents
