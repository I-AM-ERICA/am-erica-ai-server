"""Score one generator + prompt on the held-out test split, optionally against earlier runs.

    python -m scrable.evaluate --config configs/stage2_eval.json --label dpo \
        --reference outputs/stage1/test_predictions.json

The test split comes from the same data/sizes/seed as stage 1, so the reviews are identical
(checked against --reference).

To choose between several new variants without tuning on the test set, score them on the dev
split (stage 1's train + validation reviews) and run the test split only for the final pick:

    python -m scrable.evaluate --config configs/stage2_eval.json --split dev --label dpo-v1         --output-dir outputs/dev/dpo_v1
    python -m scrable.evaluate --config configs/stage2_eval.json --split dev --label sft1         --adapter outputs/stage3_sft1/adapter --output-dir outputs/dev/sft1         --reference outputs/dev/dpo_v1/dev_predictions.json Differences are reported per review with a paired bootstrap
95% confidence interval: an interval containing 0 means the gap may be chance at this n.

--split all scores every row of custom_data without splitting (e.g. the synthetic email test set,
configs/email_*.json). --reference accepts several files.
"""

import argparse
import json
import random
from pathlib import Path

from .config import Config
from .data import load_custom_reviews, load_documents
from .experiment import load_base_prompt, prepare_data, summarize
from .io import write_json
from .llm import build_llms, response_spec
from .pipeline import ScoredResponseGenerator
from .types import CRITERIA


def load_runs(path):
    """{run name: {review_id: feedback dict}} from a stage-1 or evaluate test_predictions.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = {}
    if "predictions" in data:
        runs[data.get("label", Path(path).parent.name)] = data["predictions"]
    for name in ("baseline", "optimized"):
        if name in data:
            runs[f"stage1:{name}"] = data[name]
    for comparison in data.get("comparisons", []):
        runs[f"{comparison['model']}:{comparison.get('prompt', 'base')}"] = comparison["predictions"]
    return {name: {f["review_id"]: f for f in rows} for name, rows in runs.items()}


def paired_bootstrap(a, b, samples=10000, seed=0):
    """Mean of a-b with a 95% percentile bootstrap interval over paired items."""
    if len(a) != len(b) or not a:
        raise ValueError("paired_bootstrap needs two nonempty lists of equal length")
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(samples))
    return {
        "mean": sum(diffs) / len(diffs),
        "low": means[int(0.025 * samples)],
        "high": means[int(0.975 * samples) - 1],
        "wins": sum(d > 0 for d in diffs),
        "losses": sum(d < 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
    }


def item_score(feedback, criterion):
    return feedback["score"] if criterion == "overall" else (feedback["scores"][criterion] - 1) / 4


def compare(current, reference_runs):
    """Paired comparison of the current run's feedback against each reference run."""
    ids = sorted(current)
    results = {}
    for name, run in reference_runs.items():
        if set(run) != set(ids):
            raise ValueError(f"Reference run {name} covers different test reviews; check data/sizes/seed")
        results[name] = {
            c: paired_bootstrap(
                [item_score(current[i], c) for i in ids], [item_score(run[i], c) for i in ids]
            )
            for c in ("overall", *CRITERIA)
        }
    return results


def comparison_table(label, summary, comparisons):
    lines = [
        f"\n{label}: overall {summary['overall']:.3f} (n={summary['count']}) "
        + " ".join(f"{c} {summary[c]:.3f}" for c in CRITERIA)
    ]
    for name, by_criterion in comparisons.items():
        lines.append(f"\n{label} - {name}  (95% CI; win/lose/tie)")
        for c, r in by_criterion.items():
            verdict = "chance?" if r["low"] <= 0 <= r["high"] else "clear"
            lines.append(
                f"  {c:<12} {r['mean']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}] "
                f"{r['wins']}/{r['losses']}/{r['ties']}  {verdict}"
            )
    return "\n".join(lines)


def seconds_summary(seconds):
    """Mean, 90th percentile and max of per-reply generation time."""
    ordered = sorted(seconds)
    return {
        "mean": sum(ordered) / len(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
    }


SPLITS = {
    "test": ("test",),
    "train": ("train",),
    "validation": ("validation",),
    "dev": ("train", "validation"),
    "all": (),  # every custom_data row, no split
}


def load_references(paths):
    """Merge load_runs over one path, several paths, or None."""
    if not paths:
        return {}
    runs = {}
    for path in [paths] if isinstance(paths, str) else paths:
        runs.update(load_runs(path))
    return runs


def prepare_all(config):
    if not config.custom_data or not config.documents_dir:
        raise ValueError("--split all needs custom_data and documents_dir")
    rows = load_custom_reviews(config.custom_data)
    if len({r.id for r in rows}) != len(rows):
        raise ValueError(f"Duplicate ids in {config.custom_data}")
    return {"all": rows}, load_documents(config.documents_dir, config.app_id), {}


def evaluate(
    config, label, reference=None, prepared=None, llms=None, retriever=None, split="test", describer=None
):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}")
    if prepared is None:
        prepared = prepare_all(config) if split == "all" else prepare_data(config)
    splits, documents, _ = prepared
    reviews = [r for name in SPLITS[split] or ("all",) for r in splits[name]]
    prompt = load_base_prompt(config)
    if retriever is None:
        from .rag import build_retriever

        retriever = build_retriever(config, documents)
    response_llm, judge_llm = (
        llms if llms is not None else build_llms(config, [response_spec(config), config.role("judge")])
    )
    if describer is None and config.image_model and any(r.image for r in reviews):
        from .photos import build_describer

        describer = build_describer(config)
    scorer = ScoredResponseGenerator(response_llm, judge_llm, retriever, config, describer)
    feedback = scorer(prompt, reviews, description=f"{split} split: {label}")
    speed = seconds_summary(scorer.generation_seconds)
    summary = summarize(feedback)
    current = {f.review_id: f.to_dict() for f in feedback}
    comparisons = compare(current, load_references(reference))
    judge = "{}:{}".format(*config.role("judge"))
    metrics = {
        "label": label,
        "split": split,
        "summary": summary,
        "generator": "{}:{}".format(*config.role("response")),
        "adapter": config.response_adapter,
        "prompt": prompt,
        "channel": config.channel,
        "image_model": config.image_model,
        "generation_seconds": speed,
        "evaluation": f"LLM-as-a-Judge ({judge}), not human evaluation",
        "comparisons": comparisons,
        "reference": reference,
        "config": config.to_dict(),
    }
    directory = Path(config.output_dir)
    write_json(directory / "metrics.json", metrics)
    write_json(
        directory / f"{split}_predictions.json",
        {"label": label, "predictions": [f.to_dict() for f in feedback]},
    )
    print(comparison_table(label, summary, comparisons), flush=True)
    print(
        f"\nseconds per reply: mean {speed['mean']:.2f}, p90 {speed['p90']:.2f}, max {speed['max']:.2f} "
        "(cached replies count as ~0)",
        flush=True,
    )
    print(f"\nSaved results to {directory.resolve()}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a generator on the held-out test split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--label", default="eval")
    parser.add_argument(
        "--reference",
        nargs="+",
        help="Earlier *_predictions.json file(s) on the same split to compare against",
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLITS),
        default="test",
        help="dev = stage 1 train + validation; all = every custom_data row",
    )
    parser.add_argument("--adapter", help="Override response_adapter (a LoRA adapter folder)")
    parser.add_argument("--output-dir", help="Override output_dir")
    parser.add_argument("--prompt", help="Override base_prompt_file (the generator's system prompt)")
    parser.add_argument(
        "--no-adapter", action="store_true", help="Score the base model without a LoRA adapter"
    )
    parser.add_argument(
        "--no-photos",
        action="store_true",
        help="Do not describe photos for the generator (image_model=null); the judge still sees them",
    )
    args = parser.parse_args()
    config = Config.load(args.config)
    overrides = {"image_model": None} if args.no_photos else {}
    if args.adapter and args.no_adapter:
        parser.error("--adapter and --no-adapter cannot be combined")
    if args.adapter:
        overrides["response_adapter"] = args.adapter
    if args.no_adapter:
        overrides["response_adapter"] = None
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.prompt:
        overrides["base_prompt_file"] = args.prompt
    if overrides:
        config = Config(**{**config.to_dict(), **overrides})
    evaluate(config, args.label, args.reference, split=args.split)


if __name__ == "__main__":
    main()
