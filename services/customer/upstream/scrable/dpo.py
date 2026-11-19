"""Stage 2: judge-scored preference pairs and LoRA DPO for the local response generator.

    python -m scrable.dpo pairs --config configs/stage2_dpo.json   # candidates -> judge -> pairs.jsonl
    python -m scrable.dpo train --config configs/stage2_dpo.json   # pairs.jsonl -> LoRA adapter
    python -m scrable.dpo select --config configs/stage3_generic.json  # re-pick pairs from an
                                       # earlier run's candidates.jsonl (no model or API calls)

base.channel="email" samples and judges email replies; extra_pairs adds earlier pairs.jsonl files
(e.g. the stage-2 review pairs) to training, so one adapter learns both channels.

The pairs step answers every training review with several sampled answers from the small model
(and optional teacher answers from an API model), scores them with the same LLM judge as stage 1,
and keeps (best, worst) as (chosen, rejected). The rejected answer always comes from the small
model itself, so training pushes down its own typical mistakes.

The DPO loss is implemented here directly (about 30 lines) on top of peft LoRA instead of TRL,
because current TRL releases track transformers 5 while this project pins transformers 4.
"""

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config
from .data import load_custom_reviews, load_documents
from .experiment import load_base_prompt, package_versions
from .io import write_json
from .llm import build_llms, chat_input, completion_end, response_spec
from .pipeline import ScoredResponseGenerator, generation_messages
from .types import CRITERIA

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class DPOConfig:
    # Config fields for the generator, judge, RAG and data (custom_data = training reviews,
    # documents_dir = their store documents, base_prompt_file = the prompt used in training).
    base: dict = field(default_factory=dict)
    # Pairs
    samples_per_review: int = 3
    temperature: float = 0.8
    top_p: float = 0.95
    teacher_models: list[dict] = field(default_factory=list)
    min_margin: float = 0.15  # chosen - rejected, on the 0-1 weighted judge score
    min_chosen_score: float = 0.6
    # Re-selection (select step / extra filters). rejected_strategy "lowest" = the small model's
    # worst answer; "generic" = its accurate but unspecific answer (accuracy >= generic_min_accuracy,
    # specificity <= generic_max_specificity), so the pair isolates specificity;
    # "generic_or_lowest" falls back to the worst answer when no such answer exists.
    candidates_from: str | None = None  # output_dir of an earlier pairs run
    rejected_strategy: str = "lowest"
    generic_min_accuracy: int = 4
    generic_max_specificity: int = 2
    min_chosen_specificity: int | None = None  # raw 1-5 judge score
    min_specificity_gap: int | None = None  # chosen minus rejected specificity, raw 1-5
    # Training. extra_pairs: other pairs.jsonl files trained together with this run's pairs.
    extra_pairs: list[str] = field(default_factory=list)
    beta: float = 0.1
    sft_weight: float = 0.2  # NLL on the chosen answer (RPO-style); keeps chosen likelihood from falling
    learning_rate: float = 5e-5
    epochs: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.1
    max_length: int = 2048
    eval_fraction: float = 0.1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(default_factory=lambda: list(LORA_TARGETS))
    gradient_checkpointing: bool = False

    def __post_init__(self):
        self.config = Config(**self.base)
        if self.config.backend != "local":
            raise ValueError("Stage 2 trains a local generator: base.backend must be local")
        if not self.config.custom_data or not self.config.documents_dir:
            raise ValueError("base.custom_data (training reviews) and base.documents_dir are required")
        for name in ("samples_per_review", "epochs", "grad_accum", "max_length", "lora_r", "lora_alpha"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.samples_per_review < 2 and not self.teacher_models:
            raise ValueError("Need at least 2 candidates per review: samples_per_review>=2 or teacher_models")
        for spec in self.teacher_models:
            if not isinstance(spec, dict) or set(spec) != {"backend", "model"}:
                raise ValueError('Each teacher_models item must be {"backend": ..., "model": ...}')
        if not 0 < self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be in (0, 2] and top_p in (0, 1]")
        if not 0 <= self.min_margin <= 1 or not 0 <= self.min_chosen_score <= 1:
            raise ValueError("min_margin and min_chosen_score are on the 0-1 score scale")
        if self.beta <= 0 or self.sft_weight < 0 or self.learning_rate <= 0:
            raise ValueError("beta and learning_rate must be > 0, sft_weight >= 0")
        if not 0 <= self.eval_fraction < 1 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("eval_fraction and warmup_ratio must be in [0, 1)")
        if not isinstance(self.extra_pairs, list) or not all(
            isinstance(x, str) and x.strip() for x in self.extra_pairs
        ):
            raise ValueError("extra_pairs must be a list of pairs.jsonl paths")
        if self.rejected_strategy not in ("lowest", "generic", "generic_or_lowest"):
            raise ValueError('rejected_strategy must be "lowest", "generic" or "generic_or_lowest"')
        for name in ("generic_min_accuracy", "generic_max_specificity", "min_chosen_specificity"):
            value = getattr(self, name)
            if value is not None and value not in range(1, 6):
                raise ValueError(f"{name} is a raw judge score: 1-5")
        if self.min_specificity_gap is not None and self.min_specificity_gap not in range(0, 5):
            raise ValueError("min_specificity_gap must be 0-4")

    @property
    def output_dir(self):
        return Path(self.config.output_dir)

    @property
    def pairs_file(self):
        return self.output_dir / "pairs.jsonl"

    @property
    def adapter_dir(self):
        return self.output_dir / "adapter"

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if not k.startswith("_")})  # "_comment" keys are notes

    def to_dict(self):
        return {k: v for k, v in asdict(self).items()}


# ---------------------------------------------------------------- pairs


def select_pair(
    candidates,
    min_margin,
    min_chosen_score,
    rejected_strategy="lowest",
    generic_min_accuracy=4,
    generic_max_specificity=2,
    min_chosen_specificity=None,
    min_specificity_gap=None,
):
    """Pick (chosen, rejected) from scored candidates, or (None, reason).

    chosen: highest score (ties: higher accuracy, then the small model's own answer, then order).
    rejected ("lowest"): lowest-scoring answer of the small model (ties: lower accuracy, then order).
    rejected ("generic"): the small model's least specific answer among the accurate ones
    (accuracy >= generic_min_accuracy, specificity <= generic_max_specificity).
    rejected ("generic_or_lowest"): "generic" when possible, else "lowest".
    """
    policy = [c for c in candidates if c["source"] == "policy"]
    if not policy:
        return None, "no_policy_answer"
    order = {id(c): i for i, c in enumerate(candidates)}
    chosen = max(
        candidates,
        key=lambda c: (c["score"], c["scores"]["accuracy"], c["source"] == "policy", -order[id(c)]),
    )
    generic = []
    if rejected_strategy != "lowest":
        generic = [
            c
            for c in policy
            if c is not chosen
            and c["scores"]["accuracy"] >= generic_min_accuracy
            and c["scores"]["specificity"] <= generic_max_specificity
        ]
    if rejected_strategy in ("generic", "generic_or_lowest") and generic:
        rejected = min(generic, key=lambda c: (c["scores"]["specificity"], c["score"], order[id(c)]))
    elif rejected_strategy == "generic":
        return None, "no_generic_answer"
    else:
        rejected = min(policy, key=lambda c: (c["score"], c["scores"]["accuracy"], order[id(c)]))
    if chosen is rejected or chosen["text"] == rejected["text"]:
        return None, "single_answer"
    if chosen["score"] < min_chosen_score:
        return None, "chosen_too_low"
    if min_chosen_specificity is not None and chosen["scores"]["specificity"] < min_chosen_specificity:
        return None, "chosen_not_specific"
    if (
        min_specificity_gap is not None
        and chosen["scores"]["specificity"] - rejected["scores"]["specificity"] < min_specificity_gap
    ):
        return None, "specificity_gap_too_small"
    if chosen["score"] - rejected["score"] < min_margin:
        return None, "margin_too_small"
    return (chosen, rejected), None


def selection_options(dcfg):
    return {
        "min_margin": dcfg.min_margin,
        "min_chosen_score": dcfg.min_chosen_score,
        "rejected_strategy": dcfg.rejected_strategy,
        "generic_min_accuracy": dcfg.generic_min_accuracy,
        "generic_max_specificity": dcfg.generic_max_specificity,
        "min_chosen_specificity": dcfg.min_chosen_specificity,
        "min_specificity_gap": dcfg.min_specificity_gap,
    }


PAIR_META = ("review_id", "app_id", "rating", "review", "messages")


def pair_record(meta, chosen, rejected):
    return {
        **meta,
        "chosen": chosen["text"],
        "rejected": rejected["text"],
        "chosen_source": chosen["source"],
        "chosen_score": chosen["score"],
        "rejected_score": rejected["score"],
        "chosen_scores": chosen["scores"],
        "rejected_scores": rejected["scores"],
    }


def write_pairs(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def pair_stats(pairs):
    return {
        "chosen_from": {
            s: sum(1 for p in pairs if p["chosen_source"].startswith(s)) for s in ("policy", "teacher")
        },
        "mean_margin": _mean(p["chosen_score"] - p["rejected_score"] for p in pairs),
        "mean_specificity": {
            side: _mean(p[f"{side}_scores"]["specificity"] for p in pairs) for side in ("chosen", "rejected")
        },
        "mean_accuracy": {
            side: _mean(p[f"{side}_scores"]["accuracy"] for p in pairs) for side in ("chosen", "rejected")
        },
        "rejected_accurate_but_generic": sum(
            p["rejected_scores"]["accuracy"] >= 4 and p["rejected_scores"]["specificity"] <= 2 for p in pairs
        ),
    }


def reselect_pairs(dcfg, source_dir=None):
    """Re-pick pairs from an earlier run's candidates.jsonl with this config's selection options.

    No model, retriever or API calls: the scored candidates are reused. Prompt messages come from
    candidates.jsonl (runs after this change) or from the earlier run's pairs.jsonl.
    """
    source = Path(source_dir or dcfg.candidates_from or "")
    if not (source / "candidates.jsonl").exists():
        raise ValueError(f"No candidates.jsonl in {source}; set candidates_from or pass --from")
    meta = {}
    if (source / "pairs.jsonl").exists():
        for pair in load_pairs(source / "pairs.jsonl"):
            meta[pair["review_id"]] = {k: pair.get(k) for k in PAIR_META}
    pairs, skipped, reviews = [], {}, 0
    for line in (source / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reviews += 1
        info = meta.get(row["review_id"])
        if info is None and row.get("messages"):
            info = {k: row.get(k) for k in PAIR_META}
        if info is None:
            skipped["no_prompt_messages"] = skipped.get("no_prompt_messages", 0) + 1
            continue
        pair, reason = select_pair(row["candidates"], **selection_options(dcfg))
        if pair is None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        pairs.append(pair_record(info, *pair))
    write_pairs(dcfg.pairs_file, pairs)
    summary = {
        "reviews": reviews,
        "pairs": len(pairs),
        "skipped": skipped,
        **pair_stats(pairs),
        "source": str(source),
        "selection": selection_options(dcfg),
        "config": dcfg.to_dict(),
    }
    write_json(dcfg.output_dir / "pairs_summary.json", summary)
    print(f"Wrote {len(pairs)} pairs to {dcfg.pairs_file} (skipped: {skipped})", flush=True)
    return summary


def build_pairs(dcfg, rows=None, retriever=None, policy=None, judge=None, teachers=None):
    """Write candidates.jsonl, pairs.jsonl and pairs_summary.json. Objects can be injected for tests."""
    from tqdm.auto import tqdm

    config = dcfg.config
    out = dcfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = rows if rows is not None else load_custom_reviews(config.custom_data)
    if retriever is None:
        from .rag import build_retriever

        retriever = build_retriever(config, load_documents(config.documents_dir, config.app_id))
    prompt = load_base_prompt(config)
    teacher_specs = [(t["backend"], t["model"]) for t in dcfg.teacher_models]
    if policy is None or judge is None or teachers is None:
        built = build_llms(config, [response_spec(config), config.role("judge"), *teacher_specs])
        policy, judge, teachers = built[0], built[1], built[2:]
    scorer = ScoredResponseGenerator(policy, judge, retriever, config)

    items = []
    for review in tqdm(rows, desc="Stage 2: sample small-model answers"):
        chunks = retriever.retrieve(review.review, review.app_id, config.top_k)
        messages = generation_messages(prompt, review, chunks, config.channel)
        texts = policy.sample(
            messages,
            config.response_tokens,
            dcfg.samples_per_review,
            dcfg.temperature,
            dcfg.top_p,
            config.seed,
        )
        items.append({"review": review, "chunks": chunks, "messages": messages, "texts": texts})

    def teacher_answers(item):
        return [t.complete(item["messages"], config.response_tokens) for t in teachers]

    if teachers:
        bar = tqdm(total=len(items), desc="Stage 2: teacher answers")

        def run(item):
            result = teacher_answers(item)
            bar.update(1)
            return result

        try:
            teacher_texts = scorer._map(run, items)
        finally:
            bar.close()
    else:
        teacher_texts = [[] for _ in items]

    for item, extra in zip(items, teacher_texts):
        candidates, seen = [], set()
        sources = [("policy", t) for t in item["texts"]] + [
            (f"teacher:{b}:{m}", t) for (b, m), t in zip(teacher_specs, extra)
        ]
        for source, text in sources:
            text = text.strip()
            if text and text not in seen:  # Empty or duplicate samples carry no preference signal.
                seen.add(text)
                candidates.append({"source": source, "text": text})
        item["candidates"] = candidates

    tasks = [
        (i, j, c) for i, item in enumerate(items) for j in range(len(item["candidates"])) for c in CRITERIA
    ]
    bar = tqdm(total=len(tasks), desc="Stage 2: judge")

    def judge_task(task):
        i, j, criterion = task
        item = items[i]
        result = scorer.judge_criterion(
            item["review"], item["candidates"][j]["text"], item["chunks"], criterion
        )
        bar.update(1)
        return result

    try:
        results = scorer._map(judge_task, tasks)
    finally:
        bar.close()
    for (i, j, criterion), result in zip(tasks, results):
        items[i]["candidates"][j].setdefault("judgments", {})[criterion] = result

    pairs, skipped = [], {}
    with (out / "candidates.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for item in items:
            review = item["review"]
            for candidate in item["candidates"]:
                feedback = scorer.feedback(
                    review, candidate["text"], item["chunks"], candidate.pop("judgments")
                )
                candidate.update(scores=feedback.scores, score=feedback.score, rationales=feedback.rationales)
            stream.write(
                json.dumps(
                    {
                        "review_id": review.id,
                        "app_id": review.app_id,
                        "rating": review.rating,
                        "review": review.review,
                        "messages": item["messages"],
                        "candidates": item["candidates"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            pair, reason = select_pair(item["candidates"], **selection_options(dcfg))
            if pair is None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            meta = {
                "review_id": review.id,
                "app_id": review.app_id,
                "rating": review.rating,
                "review": review.review,
                "messages": item["messages"],
            }
            pairs.append(pair_record(meta, *pair))
    write_pairs(dcfg.pairs_file, pairs)
    all_candidates = [c for item in items for c in item["candidates"]]
    by_source = {}
    for c in all_candidates:
        by_source.setdefault(c["source"].split(":")[0] if c["source"] != "policy" else "policy", []).append(
            c["score"]
        )
    summary = {
        "reviews": len(items),
        "candidates": len(all_candidates),
        "pairs": len(pairs),
        "skipped": skipped,
        "chosen_from": {
            s: sum(1 for p in pairs if p["chosen_source"].startswith(s)) for s in ("policy", "teacher")
        },
        "mean_margin": (sum(p["chosen_score"] - p["rejected_score"] for p in pairs) / len(pairs))
        if pairs
        else None,
        "mean_score_by_source": {s: sum(v) / len(v) for s, v in by_source.items()},
        "prompt": prompt,
        "config": dcfg.to_dict(),
        "judge": "{}:{}".format(*config.role("judge")),
        "llm_usage": [
            {"calls": getattr(llm, "calls", None), "cache_hits": getattr(llm, "hits", None)}
            for llm in (policy, judge, *teachers)
        ],
    }
    write_json(out / "pairs_summary.json", summary)
    print(f"Wrote {len(pairs)} pairs to {dcfg.pairs_file} (skipped: {skipped})", flush=True)
    return summary


# ---------------------------------------------------------------- training


def render_prompt(tokenizer, model_name, messages):
    return tokenizer.apply_chat_template(
        chat_input(model_name, messages), tokenize=False, add_generation_prompt=True
    )


def encode_pair(tokenizer, prompt_text, chosen, rejected, end, max_length):
    """Token ids for one pair, or None when prompt + longer answer exceeds max_length."""

    def ids(text):
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    prompt_ids = ids(prompt_text)
    chosen_ids, rejected_ids = ids(chosen + end), ids(rejected + end)
    if len(prompt_ids) + max(len(chosen_ids), len(rejected_ids)) > max_length:
        return None
    return {"prompt": prompt_ids, "chosen": chosen_ids, "rejected": rejected_ids}


def completion_logps(model, prompt_ids, completions, pad_id, device):
    """Summed log-probability of each completion given the prompt, plus completion token counts."""
    import torch
    import torch.nn.functional as F

    sequences = [prompt_ids + c for c in completions]
    width = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), width), dtype=torch.long)
    labels = torch.full((len(sequences), width), -100, dtype=torch.long)
    for row, (seq, completion) in enumerate(zip(sequences, completions)):
        input_ids[row, : len(seq)] = torch.tensor(seq)
        attention[row, : len(seq)] = 1
        labels[row, len(prompt_ids) : len(seq)] = torch.tensor(completion)
    input_ids, attention, labels = input_ids.to(device), attention.to(device), labels.to(device)
    logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = targets != -100
    token_logps = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.clamp(min=0).reshape(-1), reduction="none"
    ).view(targets.shape)
    return (token_logps * mask).sum(-1), mask.sum(-1)


def dpo_loss(
    policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, sft_weight=0.0, chosen_tokens=None
):
    """DPO (Rafailov et al., 2023) with an optional length-normalized NLL term on the chosen answer."""
    import torch.nn.functional as F

    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward)
    if sft_weight:
        loss = loss + sft_weight * (-policy_chosen / chosen_tokens)
    return loss.mean(), {
        "reward_margin": (chosen_reward - rejected_reward).mean().item(),
        "reward_accuracy": (chosen_reward > rejected_reward).float().mean().item(),
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
    }


def load_pairs(path):
    pairs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not pairs:
        raise ValueError(f"No pairs in {path}. Run `python -m scrable.dpo pairs` first.")
    for i, pair in enumerate(pairs, 1):
        if not {"messages", "chosen", "rejected"} <= set(pair):
            raise ValueError(f"Pair {i} needs messages, chosen and rejected")
    return pairs


def load_training_pairs(dcfg):
    """This run's pairs.jsonl plus every extra_pairs file, with the count from each source."""
    pairs = load_pairs(dcfg.pairs_file)
    counts = {str(dcfg.pairs_file): len(pairs)}
    for path in dcfg.extra_pairs:
        extra = load_pairs(path)
        counts[path] = len(extra)
        pairs += extra
    ids = [p["review_id"] for p in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate review_id across pairs files; is a file listed twice?")
    print(f"Training pairs by file: {counts}", flush=True)
    return pairs


def train_dpo(dcfg, pairs=None, model=None, tokenizer=None, device=None):
    """LoRA DPO. Reference log-probs come from the same model with the adapter disabled."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_linear_schedule_with_warmup, set_seed

    config = dcfg.config
    set_seed(config.seed)
    if pairs is None:
        pairs = load_training_pairs(dcfg)
    if model is None or tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        print(f"Loading {config.model} on {device} ({dtype})", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(config.model)
        model = AutoModelForCausalLM.from_pretrained(config.model, torch_dtype=dtype).to(device)
    device = device or next(model.parameters()).device.type
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=dcfg.lora_r,
        lora_alpha=dcfg.lora_alpha,
        lora_dropout=dcfg.lora_dropout,
        target_modules=dcfg.lora_targets,
    )
    model = get_peft_model(model, lora)
    if dcfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    end = completion_end(config.model, tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded, too_long = [], 0
    for pair in pairs:
        prompt_text = render_prompt(tokenizer, config.model, pair["messages"])
        item = encode_pair(tokenizer, prompt_text, pair["chosen"], pair["rejected"], end, dcfg.max_length)
        if item is None:
            too_long += 1
        else:
            encoded.append(item)
    if not encoded:
        raise ValueError("Every pair exceeds max_length; increase max_length")
    order = list(range(len(encoded)))
    random.Random(config.seed).shuffle(order)
    eval_count = int(len(encoded) * dcfg.eval_fraction) if len(encoded) >= 10 else 0
    eval_idx, train_idx = order[:eval_count], order[eval_count:]

    use_bf16 = device == "cuda" and any(p.dtype == torch.bfloat16 for p in model.parameters())

    def logps(item):
        # bf16 compute on GPU while the LoRA weights stay fp32; plain fp32 elsewhere.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            values, counts = completion_logps(
                model, item["prompt"], [item["chosen"], item["rejected"]], pad_id, device
            )
        return values, counts

    print(f"Reference log-probs for {len(encoded)} pairs (adapter disabled)...", flush=True)
    model.eval()
    reference = []
    with torch.no_grad(), model.disable_adapter():
        for item in encoded:
            reference.append(logps(item)[0].detach())

    def evaluate():
        if not eval_idx:
            return None
        model.eval()
        totals = {"loss": 0.0, "reward_accuracy": 0.0, "reward_margin": 0.0}
        with torch.no_grad():
            for i in eval_idx:
                values, counts = logps(encoded[i])
                loss, stats = dpo_loss(
                    values[0:1],
                    values[1:2],
                    reference[i][0:1],
                    reference[i][1:2],
                    dcfg.beta,
                    dcfg.sft_weight,
                    counts[0:1],
                )
                totals["loss"] += loss.item()
                totals["reward_accuracy"] += stats["reward_accuracy"]
                totals["reward_margin"] += stats["reward_margin"]
        return {k: v / len(eval_idx) for k, v in totals.items()}

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=dcfg.learning_rate, weight_decay=0.0)
    total_steps = math.ceil(len(train_idx) / dcfg.grad_accum) * dcfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * dcfg.warmup_ratio), total_steps)
    log = {"steps": [], "epochs": [], "eval_before": evaluate()}
    print(
        f"Training on {len(train_idx)} pairs ({eval_count} held out), {total_steps} optimizer steps",
        flush=True,
    )

    from tqdm.auto import tqdm

    rng = random.Random(config.seed)
    step = 0
    for epoch in range(dcfg.epochs):
        model.train()
        rng.shuffle(train_idx)
        window = []
        for n, i in enumerate(tqdm(train_idx, desc=f"DPO epoch {epoch + 1}/{dcfg.epochs}"), 1):
            values, counts = logps(encoded[i])
            loss, stats = dpo_loss(
                values[0:1],
                values[1:2],
                reference[i][0:1],
                reference[i][1:2],
                dcfg.beta,
                dcfg.sft_weight,
                counts[0:1],
            )
            (loss / dcfg.grad_accum).backward()
            window.append({"loss": loss.item(), **stats})
            if n % dcfg.grad_accum == 0 or n == len(train_idx):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                log["steps"].append(
                    {"step": step, "lr": scheduler.get_last_lr()[0]}
                    | {k: sum(w[k] for w in window) / len(window) for k in window[0]}
                )
                window = []
        result = {"epoch": epoch + 1, "train": log["steps"][-1], "eval": evaluate()}
        log["epochs"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    dcfg.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dcfg.adapter_dir)
    log.update(
        pairs=len(pairs),
        trained_pairs=len(train_idx),
        eval_pairs=eval_count,
        skipped_too_long=too_long,
        config=dcfg.to_dict(),
        versions=package_versions() | _peft_version(),
    )
    write_json(dcfg.output_dir / "train_log.json", log)
    print(f"Saved LoRA adapter to {dcfg.adapter_dir.resolve()}", flush=True)
    return log


def _peft_version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        return {"peft": version("peft")}
    except PackageNotFoundError:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Stage 2: preference pairs and LoRA DPO")
    parser.add_argument("step", choices=["pairs", "select", "train"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--from", dest="source", help="select: earlier output_dir with candidates.jsonl")
    args = parser.parse_args()
    dcfg = DPOConfig.load(args.config)
    if args.step == "select":
        summary = reselect_pairs(dcfg, args.source)
        print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, ensure_ascii=False))
    elif args.step == "pairs":
        print(
            json.dumps(
                {k: v for k, v in build_pairs(dcfg).items() if k != "config"}, indent=2, ensure_ascii=False
            )
        )
    else:
        train_dpo(dcfg)


if __name__ == "__main__":
    main()
