"""Notebook and CLI share the same complete experiment entry point."""

import argparse
import json
import platform
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import Config
from .data import download_reviews, knowledge_documents, load_custom_reviews, load_documents, split_reviews
from .io import digest, write_json
from .llm import build_llms
from .optimizer import PromptOptimizer
from .pipeline import PromptGenerator, ScoredResponseGenerator, average_score
from .prompts import BASE_PROMPT
from .rag import build_retriever
from .types import CRITERIA


def prepare_data(config):
    if config.custom_data:
        rows = load_custom_reviews(config.custom_data)
        provenance = {
            "path": config.custom_data,
            "sha256": digest([asdict(r) for r in rows]),
            "source": "user-supplied JSONL",
        }
    else:
        rows, provenance = download_reviews(config)
    splits = split_reviews(rows, config)
    documents = knowledge_documents(splits["knowledge"])
    if config.documents_dir:
        documents += load_documents(config.documents_dir, config.app_id)
    if not documents:
        raise ValueError(
            "No knowledge documents: supply documents_dir or reference replies in the knowledge split"
        )
    return splits, documents, provenance


def load_base_prompt(config):
    if not config.base_prompt_file:
        return BASE_PROMPT
    text = Path(config.base_prompt_file).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"base_prompt_file is empty: {config.base_prompt_file}")
    return text


def summarize(feedback):
    return {
        "overall": average_score(feedback),
        "count": len(feedback),
        **{
            criterion: sum((f.scores[criterion] - 1) / 4 for f in feedback) / len(feedback)
            for criterion in CRITERIA
        },
    }


def package_versions():
    result = {"python": platform.python_version()}
    for package in (
        "scrable",
        "torch",
        "transformers",
        "sentence-transformers",
        "openai",
        "anthropic",
        "numpy",
        "langchain-text-splitters",
        "azure-search-documents",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result


def run_experiment(config, prepared=None, llms=None, retriever=None, compare_llms=None):
    """Run download -> RAG -> optimize -> held-out test -> export.

    Optional object injection makes orchestration testable without model/API calls.
    No test double is used by the CLI or notebook's normal execution path.
    """
    directory = Path(config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    splits, documents, provenance = prepared if prepared is not None else prepare_data(config)
    base_prompt = load_base_prompt(config)
    manifest = {
        "config": config.to_dict(),
        "base_prompt": base_prompt,
        "provenance": provenance,
        "splits": {name: [asdict(r) for r in rows] for name, rows in splits.items()},
        "documents": [asdict(d) for d in documents],
    }
    signature = digest(manifest)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError(
                "output_dir contains a different experiment. Choose a new output_dir to avoid mixing runs."
            )
    write_json(manifest_path, {**manifest, "signature": signature, "versions": package_versions()})
    if retriever is None:
        retriever = build_retriever(config, documents)
    shared = {}
    if llms is None:
        llms = build_llms(config, shared=shared)
    compare_specs = [(spec["backend"], spec["model"]) for spec in config.compare_models]
    if compare_llms is None:
        compare_llms = build_llms(config, compare_specs, shared) if compare_specs else ()
    if len(compare_llms) != len(compare_specs):
        raise ValueError("compare_llms must match config.compare_models")
    response_llm, judge_llm, prompt_llm = llms
    scorer = ScoredResponseGenerator(response_llm, judge_llm, retriever, config)
    optimizer = PromptOptimizer(scorer, PromptGenerator(prompt_llm, config), config)
    result = optimizer.run(splits["train"], splits["validation"], base_prompt=base_prompt)
    # Test is never supplied to PromptGen, early stopping or prompt selection.
    baseline = scorer(base_prompt, splits["test"], description="Held-out test: baseline")
    optimized = scorer(result.best_prompt, splits["test"], description="Held-out test: selected prompt")
    metrics = {"baseline": summarize(baseline), "optimized": summarize(optimized)}
    metrics["delta"] = {k: metrics["optimized"][k] - metrics["baseline"][k] for k in ("overall", *CRITERIA)}
    # Reference generators (e.g. a large API model) answer the same test reviews with the base prompt
    # and are scored by the same judge: the gap the optimized small model should close.
    comparisons, comparison_predictions = [], []
    for (backend, model), llm in zip(compare_specs, compare_llms):
        feedback = ScoredResponseGenerator(llm, judge_llm, retriever, config)(
            base_prompt, splits["test"], description=f"Held-out test: {model} (base prompt)"
        )
        comparisons.append({"backend": backend, "model": model, "prompt": "base", **summarize(feedback)})
        comparison_predictions.append(
            {"backend": backend, "model": model, "predictions": [f.to_dict() for f in feedback]}
        )
    judge_backend, judge_model = config.role("judge")
    metrics.update(
        comparisons=comparisons,
        roles={
            role: dict(zip(("backend", "model"), config.role(role)))
            for role in ("response", "judge", "prompt")
        },
        best_iteration=result.best_iteration,
        stop_reason=result.stop_reason,
        selection="validation overall score; earliest prompt on ties",
        evaluation=f"LLM-as-a-Judge ({judge_backend}:{judge_model}), not human evaluation",
        rag_fingerprint=retriever.fingerprint,
        llm_usage=[
            {
                "calls": getattr(llm, "calls", None),
                "cache_hits": getattr(llm, "hits", None),
                "backend": getattr(getattr(llm, "backend", llm), "identity", None),
            }
            for llm in {id(llm): llm for llm in (*llms, *compare_llms)}.values()
        ],
    )
    write_json(directory / "metrics.json", metrics)
    write_json(
        directory / "test_predictions.json",
        {
            "baseline": [f.to_dict() for f in baseline],
            "optimized": [f.to_dict() for f in optimized],
            "comparisons": comparison_predictions,
        },
    )
    write_json(directory / "result.json", asdict(result))
    print(f"Saved results to {directory.resolve()}", flush=True)
    return {"optimization": result, "metrics": metrics, "scorer": scorer, "splits": splits}


def main():
    parser = argparse.ArgumentParser(description="SCRABLE prompt optimization (no gradient training)")
    parser.add_argument("--config", help="JSON Config file; defaults to HyperCLOVA 1.5B demo")
    parser.add_argument("--output-dir")
    parser.add_argument("--prepare-only", action="store_true", help="Download and validate data only")
    args = parser.parse_args()
    config = Config.load(args.config) if args.config else Config()
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.prepare_only:
        splits, docs, provenance = prepare_data(config)
        print(
            json.dumps(
                {
                    "splits": {k: len(v) for k, v in splits.items()},
                    "documents": len(docs),
                    "provenance": provenance,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    result = run_experiment(config)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print(score_table(result["metrics"], config))


def score_table(metrics, config):
    """Plain-text summary of held-out test scores (0-1, LLM judge)."""
    response = "{}:{}".format(*config.role("response"))
    rows = [
        (response, "base", metrics["baseline"]),
        (response, "optimized", metrics["optimized"]),
        *[(f"{c['backend']}:{c['model']}", c["prompt"], c) for c in metrics.get("comparisons", [])],
    ]
    header = f"{'generator':<60} {'prompt':<10} {'overall':>8} " + " ".join(f"{c:>11}" for c in CRITERIA)
    lines = ["", f"Held-out test (n={metrics['baseline']['count']}), {metrics['evaluation']}", header]
    for generator, prompt, scores in rows:
        values = " ".join(f"{scores[c]:>11.3f}" for c in CRITERIA)
        lines.append(f"{generator[-60:]:<60} {prompt:<10} {scores['overall']:>8.3f} {values}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
