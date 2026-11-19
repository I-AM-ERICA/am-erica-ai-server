# 역할: 검색 rollout 수집, 고정 Reference KL·GAE, Actor/Critic PPO 갱신, 평가·저장·재개를 실행합니다.
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import tempfile
import uuid
from time import perf_counter

import torch
from transformers import set_seed

from .config import Config
from .data import load_datasets, load_qa, dataset_identity
from .models import action_log_probs, load_models
from .objectives import compute_gae, policy_loss, value_loss
from .protocol import score_answer
from .retrieval import build_retriever
from .rollout import collect_rollout, prompt_ids
from .warmup import build_protocol_examples


class PPOExperiment:
    def __init__(self, config, resume_from=None):
        self.config = config.validate()
        self.output = Path(config.output_dir)
        self.step = 0
        self.protocol_warmup_steps = 0
        self.zero_signal_updates = 0
        self.rng = random.Random(config.seed)
        self.train_order = []
        self.train_examples_seen = 0
        set_seed(config.seed)
        self.documents, self.train_data, self.val_data = load_datasets(config)
        self.data_identity = dataset_identity(config, self.documents)
        self.eval_data = self.val_data
        if config.eval_max_samples and config.eval_max_samples < len(self.val_data):
            self.eval_data = random.Random(config.seed).sample(self.val_data, config.eval_max_samples)
        print(f"Data: train={len(self.train_data):,}, val={len(self.val_data):,}, "
              f"periodic_eval={len(self.eval_data):,}, documents={len(self.documents):,}", flush=True)
        self.retriever = build_retriever(config, self.documents)
        checkpoint = Path(resume_from) if resume_from else None
        if checkpoint is None and self.output.exists() and any(self.output.iterdir()):
            raise ValueError("output_dir가 비어 있지 않습니다. 새 경로 또는 resume_from을 지정하세요.")
        model_config = config
        if checkpoint:
            saved = Config(**json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))).to_dict()
            allowed_changes = {"total_updates", "output_dir", "save_every", "eval_every", "max_zero_signal_updates"}
            differences = [k for k, v in config.to_dict().items()
                           if k not in allowed_changes and saved.get(k) != v]
            if differences:
                raise ValueError(f"재개 설정이 checkpoint와 다릅니다: {differences}")
            metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("data_identity") is not None and metadata["data_identity"] != self.data_identity:
                raise ValueError("checkpoint 생성 후 데이터 내용/순서 또는 검색 인덱스가 변경되었습니다.")
            model_config = replace(config, model_revision=metadata["resolved_model_revision"])
        self.tokenizer, self.actor, self.critic = load_models(
            model_config, checkpoint / "actor" if checkpoint else None,
            checkpoint / "critic" if checkpoint else None,
        )
        self.device = next(self.actor.parameters()).device
        self.actor_optimizer = torch.optim.AdamW(
            [p for p in self.actor.parameters() if p.requires_grad], lr=config.actor_lr, weight_decay=0.0)
        self.critic_optimizer = torch.optim.AdamW(
            [p for p in self.critic.parameters() if p.requires_grad], lr=config.critic_lr, weight_decay=0.0)
        if checkpoint:
            state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
            self.critic.value_head.load_state_dict(state["value_head"])
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.step = state["step"]
            self.protocol_warmup_steps = state.get("protocol_warmup_steps", 0)
            self.zero_signal_updates = state.get("zero_signal_updates", 0)
            self.train_order = state.get("train_order", [])
            self.train_examples_seen = state.get("train_examples_seen", self.step * config.rollout_batch_size)
            self.rng.setstate(state["python_rng"])
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "config.json").write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        # Validate the exact same prompt used by generation and supervised warmup.
        for index, row in enumerate(self.train_data + self.val_data, 1):
            prompt_ids(self.tokenizer, row["question"], config)
            if index % 10000 == 0:
                print(f"Validated {index:,} prompt lengths", flush=True)

    def warmup_protocol(self, steps=20, learning_rate=1e-4):
        """Optional train-only SFT before PPO. Never substitutes teacher answers into PPO/evaluation."""
        if self.step or self.actor_optimizer.state:
            raise ValueError("warmup은 PPO 시작 전 새 실험에서만 실행하세요.")
        if not isinstance(steps, int) or steps < 1 or not 0 < learning_rate < float("inf"):
            raise ValueError("warmup steps와 learning_rate는 유한한 양수여야 합니다.")
        examples = build_protocol_examples(self.tokenizer, self.retriever, self.train_data, self.config)
        parameters = [p for p in self.actor.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
        history, order = [], []
        self.actor.train()
        for _ in range(steps):
            if not order:
                order = list(range(len(examples)))
                self.rng.shuffle(order)
            example = examples[order.pop()]
            ids, positions = example.tensors(self.device)
            optimizer.zero_grad(set_to_none=True)
            loss = -action_log_probs(self.actor, ids, positions).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Protocol warmup loss is not finite")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm,
                                                 error_if_nonfinite=True)
            if float(norm) == 0:
                raise RuntimeError("형식 warmup gradient가 0입니다. LoRA와 gradient 설정을 확인하세요.")
            optimizer.step()
            self.protocol_warmup_steps += 1
            metrics = {"phase": "protocol_sft", "warmup_step": self.protocol_warmup_steps,
                       "loss": float(loss.detach()), "actor_grad_norm": float(norm),
                       "learning_rate": learning_rate}
            self._append("warmup.jsonl", metrics)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            history.append(metrics)
        optimizer.zero_grad(set_to_none=True)
        self.actor.eval()
        self.zero_signal_updates = 0
        return history

    @torch.no_grad()
    def diagnose_rollouts(self, count=4, sample=True):
        """Inspect TRAIN examples without running optimizer steps or consuming training RNG state."""
        if count < 1:
            raise ValueError("count must be positive")
        rng = self.rng.getstate()
        records = []
        try:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                for index in range(count):
                    row = self.train_data[index % len(self.train_data)]
                    trajectory = self.rollout(row["question"], sample=sample)
                    records.append({"id": row["id"], "finish_reason": trajectory.finish_reason,
                                    "action_tokens": sum(trajectory.action_mask),
                                    "events": trajectory.events, **self._score(trajectory, row)})
        finally:
            self.rng.setstate(rng)
        summary = {"count": count, "finish_reasons": dict(Counter(r["finish_reason"] for r in records)),
                   **{key: sum(r[key] for r in records) / count
                      for key in ("reward", "exact_match", "valid_answer", "searches")}}
        self._append("diagnostics.jsonl", {"step": self.step, "sample": sample,
                                          "summary": summary, "examples": records})
        return {"summary": summary, "examples": records}

    def rollout(self, question, sample=False):
        return collect_rollout(self.actor, self.tokenizer, self.retriever, question, self.config, sample)

    def _score(self, trajectory, row):
        return score_answer(trajectory.answer, row["answers"], trajectory.searches,
                            self.config.format_reward, self.config.search_cost,
                            self.config.answer_normalization)

    def _next_train_row(self):
        if self.config.train_sampling == "shuffle":
            if not self.train_order:
                self.train_order = list(range(len(self.train_data)))
                self.rng.shuffle(self.train_order)
            row = self.train_data[self.train_order.pop()]
        else:
            row = self.rng.choice(self.train_data)
        self.train_examples_seen += 1
        return row

    @torch.no_grad()
    def _collect_batch(self):
        self.actor.eval()
        self.critic.eval()
        batch = []
        for _ in range(self.config.rollout_batch_size):
            row = self._next_train_row()
            trajectory = self.rollout(row["question"], sample=True)
            ids, positions = trajectory.tensors(self.device)
            old_logp = action_log_probs(self.actor, ids, positions)
            # 초기 base는 frozen이고 LoRA만 학습하므로 adapter 비활성화가 고정 reference입니다.
            with self.actor.disable_adapter():
                ref_logp = action_log_probs(self.actor, ids, positions)
            old_values = self.critic(ids, positions)
            score = self._score(trajectory, row)
            rewards = -self.config.kl_coef * (old_logp - ref_logp)
            rewards[-1] += score["reward"]
            advantages, returns = compute_gae(rewards, old_values, self.config.gamma, self.config.gae_lambda)
            batch.append(dict(trajectory=trajectory, row=row, score=score,
                              old_logp=old_logp.cpu(), old_values=old_values.cpu(),
                              advantages=advantages.cpu(), returns=returns.cpu(),
                              raw_advantages=advantages.cpu().clone(),
                              reference_kl=float((old_logp - ref_logp).mean())))
        # 모든 생성 action token에 대해 정규화하고 검색 관측 token은 포함하지 않습니다.
        all_advantages = torch.cat([b["advantages"] for b in batch])
        mean, std = all_advantages.mean(), all_advantages.std(unbiased=False).clamp_min(1e-8)
        for item in batch:
            item["advantages"] = (item["advantages"] - mean) / std
        return batch

    def _update(self, batch):
        total_tokens = sum(len(b["old_logp"]) for b in batch)
        metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0,
                   "actor_grad_norm": 0.0, "critic_grad_norm": 0.0}
        self.actor.train()
        self.critic.train()
        for _ in range(self.config.ppo_epochs):
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            # trajectory 1개씩 순전파/역전파하여 variable-length minibatch를 누적합니다.
            for item in batch:
                ids, positions = item["trajectory"].tensors(self.device)
                logp = action_log_probs(self.actor, ids, positions)
                loss, extra = policy_loss(logp, item["old_logp"].to(self.device),
                                          item["advantages"].to(self.device), self.config.clip_range)
                actor_loss = loss.sum() / total_tokens
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Actor loss is not finite")
                actor_loss.backward()
                # Actor graph를 해제한 뒤 Critic graph를 만듭니다.
                actor_loss_value = actor_loss.detach().item()
                del logp, loss, actor_loss
                values = self.critic(ids, positions)
                critic_loss = value_loss(values, item["old_values"].to(self.device),
                                         item["returns"].to(self.device), self.config.value_clip_range).sum() / total_tokens
                if not torch.isfinite(critic_loss):
                    raise FloatingPointError("Critic loss is not finite")
                critic_loss.backward()
                metrics["actor_loss"] += actor_loss_value / self.config.ppo_epochs
                metrics["critic_loss"] += critic_loss.detach().item() / self.config.ppo_epochs
                weight = len(positions) / total_tokens / self.config.ppo_epochs
                for name, value in extra.items():
                    metrics[name] += weight * value
            for name, model in (("actor", self.actor), ("critic", self.critic)):
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    self.config.max_grad_norm, error_if_nonfinite=True)
                metrics[name + "_grad_norm"] += float(norm) / self.config.ppo_epochs
            for optimizer in (self.actor_optimizer, self.critic_optimizer):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        return metrics

    @torch.no_grad()
    def evaluate(self, full=False, split="val"):
        if split not in {"val", "test"}:
            raise ValueError("Evaluation split must be val or test")
        if split == "test":
            if not self.config.test_path:
                raise ValueError("test_path is not configured")
            rows = load_qa(self.config.test_path)
        else:
            rows = self.val_data if full else self.eval_data
        prefix = "test_" if split == "test" else "eval_"
        totals = Counter()
        by_source, source_counts = {}, Counter()
        evaluation_id = uuid.uuid4().hex
        for index, row in enumerate(rows, 1):
            trajectory = self.rollout(row["question"], sample=False)
            score = self._score(trajectory, row)
            totals.update(score)
            source = row.get("data_source", "unknown")
            by_source.setdefault(source, Counter()).update(score)
            source_counts[source] += 1
            self._append("evaluation_examples.jsonl", {
                "evaluation_id": evaluation_id, "step": self.step, "split": split,
                "id": row["id"], "data_source": source, "question": row["question"],
                "answer": trajectory.answer, "expected": row["answers"],
                "finish_reason": trajectory.finish_reason, "events": trajectory.events, **score})
            if index % 50 == 0:
                print(f"Evaluation {split}: {index:,}/{len(rows):,}", flush=True)
        metrics = {prefix + key: total / len(rows) for key, total in totals.items()}
        metrics[prefix + "count"] = len(rows)
        self._append("evaluation.jsonl", {"step": self.step, "protocol_warmup_steps": self.protocol_warmup_steps,
                                          "split": split, "full": split == "test" or full or len(rows) == len(self.val_data),
                                          "evaluation_id": evaluation_id,
                                          "by_source": {source: {"count": source_counts[source], **{
                                              key: total / source_counts[source] for key, total in values.items()}}
                                                        for source, values in by_source.items()}, **metrics})
        return metrics

    def _append(self, name, row):
        with (self.output / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save(self):
        destination = self.output / f"checkpoint-{self.step:05d}"
        if destination.exists():
            raise FileExistsError(f"기존 checkpoint를 덮어쓰지 않습니다: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=self.output))
        self.actor.save_pretrained(temporary / "actor")
        self.critic.backbone.save_pretrained(temporary / "critic")
        self.tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save({"step": self.step, "value_head": self.critic.value_head.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                    "protocol_warmup_steps": self.protocol_warmup_steps,
                    "zero_signal_updates": self.zero_signal_updates,
                    "train_order": self.train_order, "train_examples_seen": self.train_examples_seen,
                    "python_rng": self.rng.getstate(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()}, temporary / "training_state.pt")
        (temporary / "config.json").write_text(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        import transformers, peft
        metadata = {"resolved_model_revision": self.actor.config._commit_hash or self.config.model_revision,
                    "data_identity": self.data_identity,
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "peft": peft.__version__, "gpu": torch.cuda.get_device_name(),
                    "reference": "frozen base with actor adapter disabled (also after optional protocol SFT)",
                    "protocol_warmup_steps": self.protocol_warmup_steps}
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
        return destination

    def train(self):
        history = []
        stop_file = self.output / "STOP_AFTER_STEP"
        for step in range(self.step + 1, self.config.total_updates + 1):
            if stop_file.exists():
                if not (self.output / f"checkpoint-{self.step:05d}").exists():
                    self.save()
                print(f"Stopped at saved update {self.step}: {stop_file}", flush=True)
                break
            started = perf_counter()
            search_before = getattr(self.retriever, "search_seconds", 0.0)
            hits_before = getattr(self.retriever, "cache_hits", 0)
            batch = self._collect_batch()
            collected = perf_counter()
            metrics = self._update(batch)
            updated = perf_counter()
            self.step = step
            metrics.update(collect_seconds=collected - started, update_seconds=updated - collected,
                           search_seconds=getattr(self.retriever, "search_seconds", 0.0) - search_before,
                           search_cache_hits=getattr(self.retriever, "cache_hits", 0) - hits_before)
            metrics.update({key: sum(b["score"][key] for b in batch) / len(batch)
                            for key in batch[0]["score"]})
            metrics["reference_kl_sampled"] = sum(b["reference_kl"] for b in batch) / len(batch)
            raw_advantages = torch.cat([b["raw_advantages"] for b in batch])
            metrics.update({"advantage_std": float(raw_advantages.std(unbiased=False)),
                            "advantage_abs_mean": float(torch.cat([b["advantages"] for b in batch]).abs().mean()),
                            "action_tokens": sum(len(b["old_logp"]) for b in batch),
                            "finish_reasons": dict(Counter(b["trajectory"].finish_reason for b in batch))})
            zero_signal = metrics["actor_grad_norm"] == 0 and metrics["critic_grad_norm"] == 0
            self.zero_signal_updates = self.zero_signal_updates + 1 if zero_signal else 0
            metrics["zero_signal_updates"] = self.zero_signal_updates
            metrics["step"] = step
            metrics["train_examples_seen"] = self.train_examples_seen
            metrics["train_passes"] = self.train_examples_seen / len(self.train_data)
            stop_requested = stop_file.exists()
            # Persist before potentially long validation; never save a failed
            # zero-signal run as a successful periodic checkpoint.
            checkpoint_due = step % self.config.save_every == 0 or step == self.config.total_updates
            if (checkpoint_due or stop_requested) and self.zero_signal_updates < self.config.max_zero_signal_updates:
                self.save()
            if step % self.config.eval_every == 0 and not stop_requested:
                evaluation_started = perf_counter()
                metrics.update(self.evaluate())
                metrics["evaluation_seconds"] = perf_counter() - evaluation_started
            metrics["step_seconds"] = perf_counter() - started
            self._append("metrics.jsonl", metrics)
            # Log every rollout: the previous first-only trace hid rare valid/rewarding samples.
            for index, example in enumerate(batch):
                self._append("rollouts.jsonl", {"step": step, "batch_index": index,
                                               "id": example["row"]["id"],
                                               "finish_reason": example["trajectory"].finish_reason,
                                               "events": example["trajectory"].events,
                                               **example["score"]})
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            history.append(metrics)
            if self.zero_signal_updates >= self.config.max_zero_signal_updates:
                raise RuntimeError(
                    f"{self.zero_signal_updates}회 연속 Actor/Critic gradient가 모두 0입니다. "
                    f"학습 신호 없이 진행하는 것을 중단했습니다. {self.output / 'rollouts.jsonl'}에서 "
                    "생성문과 종료 사유를 확인하세요. 검색/답변 형식 학습 또는 보상·정답 매칭 점검이 필요합니다. "
                    "저장된 파일은 유지합니다. 형식 warmup은 새 실험에서 수행하세요.")
            if stop_requested:
                print(f"Stopped at saved update {self.step}: {stop_file}", flush=True)
                break
        return history
