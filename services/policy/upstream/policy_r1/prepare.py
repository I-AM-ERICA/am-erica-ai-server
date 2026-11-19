"""Convert Search-R1 QA Parquet into deterministic, disjoint JSONL splits."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile

from .data import question_key


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_parquet_qa(path, split):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = [c for c in ("id", "question", "golden_answers", "answers", "data_source", "reward_model")
               if c in parquet.schema_arrow.names]
    if "question" not in columns:
        raise ValueError(f"{path}: question column is required; prompt is not used as a question")
    unique, duplicates, count = {}, 0, 0
    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
        for raw in batch.to_pylist():
            index = count
            count += 1
            question = raw["question"]
            answers = raw.get("golden_answers") or raw.get("answers")
            if not answers:
                ground_truth = (raw.get("reward_model") or {}).get("ground_truth", {})
                answers = ground_truth.get("target") if isinstance(ground_truth, dict) else ground_truth
            if isinstance(answers, str):
                answers = [answers]
            if not isinstance(question, str) or not question.strip() or not question_key(question):
                raise ValueError(f"{path} row {index}: invalid question")
            if not isinstance(answers, list) or not answers or any(
                not isinstance(a, str) or not a.strip() for a in answers
            ):
                raise ValueError(f"{path} row {index}: invalid answers")
            source = str(raw.get("data_source") or "unknown")
            row = {"id": f"{source}:{split}:{index}", "source_id": str(raw.get("id", index)),
                   "data_source": source, "source_split": split, "question": question.strip(),
                   "answers": list(dict.fromkeys(a.strip() for a in answers))}
            key = question_key(question)
            if key in unique:
                duplicates += 1
                # Keep all aliases, but never merge answers across train and test.
                unique[key]["answers"] = list(dict.fromkeys(unique[key]["answers"] + row["answers"]))
            else:
                unique[key] = row
    if not unique:
        raise ValueError(f"{path}: no QA rows")
    return unique, {"raw_rows": count, "duplicate_questions_merged": duplicates}


def prepare_qa(train_path, test_path, output_dir, validation_size=2000, seed=42):
    if validation_size < 1:
        raise ValueError("validation_size must be positive")
    output = Path(output_dir)
    inputs = {"version": 1, "train_sha256": file_sha256(train_path),
              "test_sha256": file_sha256(test_path), "validation_size": validation_size, "seed": seed}
    if output.exists():
        manifest_path = output / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("inputs") == inputs and all(
                (output / name).is_file() and file_sha256(output / name) == digest
                for name, digest in manifest["output_sha256"].items()
            ):
                return manifest
        raise FileExistsError(f"Existing prepared data differs or is incomplete: {output}. Use a new output directory.")
    train, train_stats = read_parquet_qa(train_path, "train")
    test, test_stats = read_parquet_qa(test_path, "test")
    overlap = train.keys() & test.keys()
    for key in overlap:
        del train[key]
    rows = list(train.values())
    if len(rows) <= validation_size:
        raise ValueError(f"Only {len(rows)} disjoint training questions; validation_size={validation_size}")
    # Seeded random split; all duplicate questions were grouped before splitting.
    random.Random(seed).shuffle(rows)
    splits = {"train": rows[validation_size:], "val": rows[:validation_size], "test": list(test.values())}
    manifest = {"inputs": inputs, "source_files": {"train": str(train_path), "test": str(test_path)},
                "source_stats": {"train": train_stats, "test": test_stats},
                "train_test_overlap_removed_from_train": len(overlap),
                "counts": {name: len(items) for name, items in splits.items()},
                "data_sources": {name: dict(Counter(r["data_source"] for r in items))
                                 for name, items in splits.items()}, "output_sha256": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".qa-", dir=output.parent))
    try:
        for split, items in splits.items():
            path = temporary / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in items:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest["output_sha256"][path.name] = file_sha256(path)
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest
