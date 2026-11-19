"""Save committed weights after a verified interrupt during rollout collection.

For a legacy live kernel only. New runs should use STOP_AFTER_STEP instead.
The incomplete batch is replayed; its already-consumed random draws cannot be
recovered, so this is not a bit-for-bit replay of an uninterrupted run.
"""
import json
import sys
from pathlib import Path


def save_interrupted_collection(experiment, exception=None, traceback=None):
    exception = exception if exception is not None else getattr(sys, "last_value", None)
    traceback = traceback if traceback is not None else getattr(sys, "last_traceback", None)
    if not isinstance(exception, KeyboardInterrupt) or traceback is None:
        raise RuntimeError("A KeyboardInterrupt traceback from the training cell is required")
    if experiment.config.train_sampling != "shuffle":
        raise RuntimeError("Legacy collection recovery currently requires shuffle sampling")
    frames = []
    while traceback:
        frames.append(traceback.tb_frame)
        traceback = traceback.tb_next
    owned = [f for f in frames if f.f_locals.get("self") is experiment]
    collection = [f for f in owned if f.f_code.co_name == "_collect_batch"]
    if len(collection) != 1 or any(f.f_code.co_name == "_update" for f in owned):
        raise RuntimeError("Not interrupted in rollout collection; refusing to save potentially partial weights")
    if not any(f.f_code.co_name == "search" and
               Path(f.f_code.co_filename).name == "sqlite_search.py" for f in frames):
        raise RuntimeError("Not interrupted in SQLite search; no state changed")
    if getattr(experiment, "_collection_recovered", False):
        raise RuntimeError("This interrupted batch has already been recovered")
    scope = collection[0].f_locals
    batch, row = scope.get("batch"), scope.get("row")
    if not isinstance(batch, list) or row is None:
        raise RuntimeError("Cannot identify the incomplete batch")
    consumed = [item["row"] for item in batch] + [row]
    if not 1 <= len(consumed) <= experiment.config.rollout_batch_size:
        raise RuntimeError("Unexpected number of consumed rows")
    # A completed step must have a matching durable metrics record.
    completed = None
    metrics_path = experiment.output / "metrics.jsonl"
    if metrics_path.exists():
        with metrics_path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record["step"] == experiment.step:
                    completed = record["train_examples_seen"]
    if experiment.step == 0:
        completed = 0
    if completed is None or experiment.train_examples_seen != completed + len(consumed):
        raise RuntimeError("Sampler counters do not match the last completed update; no state changed")
    destination = experiment.output / f"checkpoint-{experiment.step:05d}"
    if destination.exists():
        raise FileExistsError(f"Use the existing completed checkpoint: {destination}")
    by_identity = {id(item): index for index, item in enumerate(experiment.train_data)}
    indices = [by_identity[id(item)] for item in consumed]
    old_order = list(experiment.train_order)
    old_seen = experiment.train_examples_seen
    old_save_every = experiment.config.save_every
    try:
        if experiment.config.train_sampling == "shuffle":
            # pop() yields the interrupted rows again in their original order,
            # including when collection crossed a shuffled-epoch boundary.
            experiment.train_order.extend(reversed(indices))
        experiment.train_examples_seen = completed
        experiment.config.save_every = 10
        destination = experiment.save()
    except BaseException:
        experiment.train_order = old_order
        experiment.train_examples_seen = old_seen
        experiment.config.save_every = old_save_every
        raise
    experiment._collection_recovered = True
    note = {"step": experiment.step, "replayed_rows": len(consumed),
            "reason": "KeyboardInterrupt verified in SQLite rollout search",
            "rng": "current RNG retained; incomplete rollout will be resampled"}
    (destination / "recovery.json").write_text(json.dumps(note, indent=2))
    (experiment.output / "RECOVERED_CHECKPOINT").write_text(str(destination.resolve()) + "\n")
    print(f"Saved completed PPO update {experiment.step}: {destination}", flush=True)
    print(f"Incomplete batch rows to replay: {len(consumed)}; save_every=10", flush=True)
    return destination
