import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

BACKENDS = {"local", "openai", "anthropic"}
EFFORTS = {"low", "medium", "high", "xhigh", "max"}


@dataclass
class Config:
    # Response generator backend/model. Judge and prompt generator default to the same values.
    backend: str = "local"
    model: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    # Stage 1: a strong API judge/prompt generator can teach a small local response generator.
    judge_backend: str | None = None
    judge_model: str | None = None
    prompt_backend: str | None = None
    prompt_model: str | None = None
    # Extra generators scored on the held-out test set with the base prompt, e.g. a large API model.
    compare_models: list[dict] = field(default_factory=list)
    # Parallel judge requests. Use >1 only for API judges; a local model is not thread-safe.
    judge_workers: int = 1
    # API models can spend output tokens on reasoning. Role token budgets act as a floor of this size.
    api_min_output_tokens: int = 16000
    # Optional Claude effort (low|medium|high|xhigh|max). None keeps the model default.
    anthropic_effort: str | None = None
    # Optional UTF-8 text file replacing the built-in base prompt (e.g. store-review domain).
    base_prompt_file: str | None = None
    # Stage 2: LoRA adapter folder (from `python -m scrable.dpo train`) applied to the local generator.
    response_adapter: str | None = None
    device: str = "auto"
    # Optional local CPU inference compression, applied after merging a LoRA adapter.
    cpu_quantization: str | None = None
    seed: int = 42
    train_size: int = 10
    validation_size: int = 4
    test_size: int = 4
    knowledge_size: int = 64  # Past replies reserved for RAG. 0 is allowed when documents_dir is set.
    max_iterations: int = 2  # Number of revisions, excluding baseline (iteration 0).
    threshold: float = 0.95
    lowest_fraction: float = 0.30
    random_fraction: float = 0.10
    dataset_revision: str = "49068c54a4551304208aea80749f1a4d5ab893d2"
    app_id: str = "com.ubercab.eats"
    download_limit: int = 3000  # Eligible, unique reviews from a bounded source prefix.
    max_scan_lines: int = 100000
    data_dir: str = "data"
    output_dir: str = "outputs/demo"
    custom_data: str | None = None
    documents_dir: str | None = None
    rag_backend: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    azure_embedding_model: str = "text-embedding-ada-002"
    azure_index_prefix: str = "scrable"
    chunk_tokens: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    context_tokens: int = 8192
    response_tokens: int = 256
    judge_tokens: int = 256
    prompt_tokens: int = 768
    json_attempts: int = 3
    # "review": store review replies. "email": customer email replies (judge wording and the
    # generator input key change; criteria and weights stay the same).
    channel: str = "review"
    # Photo reviews (plan A): local vision model that describes each review's photo for the
    # generator, e.g. "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B". None = photos unused
    # by the generator (the API judge still sees them).
    image_model: str | None = None

    def __post_init__(self):
        if self.cpu_quantization not in (None, "int8"):
            raise ValueError("cpu_quantization must be null or int8")
        if self.cpu_quantization and (self.device != "cpu" or self.backend != "local"):
            raise ValueError("cpu_quantization requires backend=local and device=cpu")
        for name in ("backend", "judge_backend", "prompt_backend"):
            value = getattr(self, name)
            if value is not None and value not in BACKENDS:
                raise ValueError(f"{name} must be one of {sorted(BACKENDS)}")
        if self.channel not in {"review", "email"}:
            raise ValueError("channel must be review or email")
        if self.image_model is not None and (
            "hyperclovax" not in self.image_model.lower() or "vision" not in self.image_model.lower()
        ):
            raise ValueError("image_model must be a HyperCLOVAX-SEED-Vision model or null")
        if self.rag_backend not in {"local", "azure"}:
            raise ValueError("rag_backend must be local or azure")
        for name in (
            "train_size",
            "validation_size",
            "test_size",
            "download_limit",
            "max_scan_lines",
            "top_k",
            "chunk_tokens",
            "context_tokens",
            "response_tokens",
            "judge_tokens",
            "prompt_tokens",
            "json_attempts",
            "judge_workers",
            "api_min_output_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.knowledge_size) is not int or self.knowledge_size < 0:
            raise ValueError("knowledge_size must be >= 0")
        if self.knowledge_size == 0 and not self.documents_dir:
            raise ValueError("knowledge_size=0 requires documents_dir for RAG documents")
        if type(self.max_iterations) is not int or self.max_iterations < 0:
            raise ValueError("max_iterations must be >= 0")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold is normalized to [0, 1]")
        if not 0 < self.lowest_fraction <= 1 or not 0 <= self.random_fraction <= 1:
            raise ValueError("invalid selection fractions")
        if self.lowest_fraction + self.random_fraction > 1:
            raise ValueError("selection fractions must sum to <= 1")
        if not 0 <= self.chunk_overlap < self.chunk_tokens:
            raise ValueError("chunk_overlap must be smaller than chunk_tokens")
        if max(self.response_tokens, self.judge_tokens, self.prompt_tokens) >= self.context_tokens:
            raise ValueError("output budget must be smaller than context_tokens")
        if not self.model.strip() or not self.app_id.strip():
            raise ValueError("model and app_id cannot be empty")
        for role, backend, model in (
            ("judge", self.judge_backend, self.judge_model),
            ("prompt", self.prompt_backend, self.prompt_model),
        ):
            if backend not in (None, self.backend) and not (model or "").strip():
                raise ValueError(f"{role}_model is required when {role}_backend differs from backend")
        if self.judge_workers > 1 and self.role("judge")[0] == "local":
            raise ValueError("judge_workers > 1 requires an API judge_backend (openai or anthropic)")
        if self.response_adapter is not None and self.backend != "local":
            raise ValueError("response_adapter requires backend=local")
        if self.anthropic_effort is not None and self.anthropic_effort not in EFFORTS:
            raise ValueError(f"anthropic_effort must be one of {sorted(EFFORTS)} or null")
        if not isinstance(self.compare_models, list):
            raise ValueError("compare_models must be a list")
        for spec in self.compare_models:
            if (
                not isinstance(spec, dict)
                or set(spec) != {"backend", "model"}
                or spec["backend"] not in BACKENDS
                or not isinstance(spec["model"], str)
                or not spec["model"].strip()
            ):
                raise ValueError(
                    'Each compare_models item must be {"backend": local|openai|anthropic, "model": "..."}'
                )

    def role(self, name):
        """Return (backend, model) for the response, judge or prompt role."""
        if name == "response":
            return self.backend, self.model
        if name == "judge":
            return self.judge_backend or self.backend, self.judge_model or self.model
        if name == "prompt":
            return self.prompt_backend or self.backend, self.prompt_model or self.model
        raise ValueError(f"Unknown role: {name}")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if not k.startswith("_")})  # "_comment" keys are notes

    def to_dict(self):
        return asdict(self)
