# 역할: 모델·검색·PPO 설정을 읽고 잘못된 실험 설정을 조기에 검증합니다.
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    model_name: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    model_revision: str = "main"
    seed: int = 42
    load_in_4bit: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    gradient_checkpointing: bool = True
    corpus_path: str = "data/demo/corpus.jsonl"
    train_path: str = "data/demo/train.jsonl"
    val_path: str = "data/demo/val.jsonl"
    test_path: str = ""
    allow_demo: bool = True
    retriever: str = "bm25"
    dense_model: str = "intfloat/multilingual-e5-small"
    prompt_language: str = "ko"
    answer_normalization: str = "compact"
    train_sampling: str = "replacement"
    eval_max_samples: int = 0  # 0 = full validation; otherwise fixed seeded subset
    warmup_max_examples: int = 0
    warmup_max_candidates: int = 0
    warmup_require_evidence: bool = True
    warmup_skip_unusable: bool = False
    top_k: int = 3
    max_seq_length: int = 1536
    max_prompt_tokens: int = 512
    max_new_tokens: int = 192
    max_observation_tokens: int = 192
    observation_strategy: str = "concat"
    max_searches: int = 2
    rollout_batch_size: int = 4
    total_updates: int = 10
    ppo_epochs: int = 2
    actor_lr: float = 0.00001
    critic_lr: float = 0.00005
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    kl_coef: float = 0.01
    max_grad_norm: float = 1.0
    format_reward: float = 0.0
    search_cost: float = 0.0
    eval_every: int = 5
    save_every: int = 5
    max_zero_signal_updates: int = 3
    output_dir: str = "outputs/first_experiment"

    def validate(self):
        if self.observation_strategy not in {"concat", "balanced_v1", "focused_v1"}:
            raise ValueError("Invalid observation_strategy")
        for key in ("lora_rank", "lora_alpha", "top_k", "max_seq_length",
                    "max_prompt_tokens", "max_new_tokens", "max_observation_tokens",
                    "rollout_batch_size", "total_updates", "ppo_epochs", "eval_every", "save_every",
                    "max_zero_signal_updates"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.max_searches < 0 or self.max_prompt_tokens >= self.max_seq_length:
            raise ValueError("검색 횟수 또는 prompt/sequence 길이 설정을 확인하세요.")
        if self.retriever not in {"bm25", "dense", "sqlite_bm25"}:
            raise ValueError("retriever must be bm25, dense or sqlite_bm25")
        if self.prompt_language not in {"ko", "en"}:
            raise ValueError("prompt_language must be ko or en")
        if self.answer_normalization not in {"compact", "search_r1"}:
            raise ValueError("answer_normalization must be compact or search_r1")
        if self.train_sampling not in {"replacement", "shuffle"}:
            raise ValueError("train_sampling must be replacement or shuffle")
        for key in ("eval_max_samples", "warmup_max_examples", "warmup_max_candidates"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma/gae_lambda must be in [0, 1]")
        if min(self.actor_lr, self.critic_lr, self.max_grad_norm) <= 0:
            raise ValueError("학습률과 gradient clipping 값은 양수여야 합니다.")
        if min(self.kl_coef, self.search_cost, self.format_reward,
               self.clip_range, self.value_clip_range) < 0:
            raise ValueError("보상·clipping 계수는 음수일 수 없습니다.")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path="configs/first_experiment.yaml"):
    with Path(path).open(encoding="utf-8") as handle:
        return Config(**yaml.safe_load(handle)).validate()
