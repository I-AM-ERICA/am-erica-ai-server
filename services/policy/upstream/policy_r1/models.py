# 역할: 독립 Actor/Critic LoRA 모델과 Value Head, 고정 Reference log-prob 계산을 제공합니다.
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def prepare_backbone(base, config, task_type):
    base.config.use_cache = False
    if config.load_in_4bit:
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif config.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type=task_type,
    ))
    # PPO rollout과 재계산의 분포를 일치시킵니다. train mode에서도 dropout은 0입니다.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
    return model


class ValueModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(backbone.config.hidden_size, 1, dtype=torch.float32)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.value_head.to(next(backbone.parameters()).device)

    def forward(self, ids, positions):
        hidden = self.backbone(input_ids=ids, attention_mask=torch.ones_like(ids),
                               use_cache=False, return_dict=True).last_hidden_state
        # token t를 선택하기 직전 상태 h[t-1]의 가치를 예측합니다.
        return self.value_head(hidden[0, positions - 1].float()).squeeze(-1)


def load_models(config, actor_adapter=None, critic_adapter=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 없습니다. Naver Cloud GPU 서버의 Jupyter 커널을 선택하세요.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    kwargs = dict(revision=config.model_revision, torch_dtype=dtype,
                  device_map={"": torch.cuda.current_device()}, attn_implementation="sdpa")
    if config.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise ValueError("이 실험에는 모델의 chat_template이 필요합니다.")

    def make(cls, task, adapter):
        base = cls.from_pretrained(config.model_name, **kwargs)
        if adapter is None:
            return prepare_backbone(base, config, task)
        # 재개 시에도 동일한 초기화 후 adapter tensor를 로드합니다.
        model = prepare_backbone(base, config, task)
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        result = set_peft_model_state_dict(model, load_file(str(adapter / "adapter_model.safetensors")))
        if result.unexpected_keys:
            raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")
        return model

    actor = make(AutoModelForCausalLM, "CAUSAL_LM", actor_adapter)
    # AutoModel은 언어모델 출력 head 없이 같은 HyperCLOVA backbone을 별도로 로드합니다.
    critic = ValueModel(make(AutoModel, "FEATURE_EXTRACTION", critic_adapter))
    if config.max_seq_length > actor.config.max_position_embeddings:
        raise ValueError("max_seq_length가 모델의 context 한도를 초과합니다.")
    actor.print_trainable_parameters()
    critic.backbone.print_trainable_parameters()
    return tokenizer, actor, critic


def action_log_probs(actor, ids, positions):
    # 해당 모델은 공식 카드의 llama architecture입니다. action 위치만 vocab에 투영해 메모리를 줄입니다.
    base = actor.get_base_model()
    hidden = base.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                        use_cache=False, return_dict=True).last_hidden_state[0, positions - 1]
    targets = ids[0, positions]

    def logprob_chunk(states, labels):
        logits = base.lm_head(states).float()
        return -nn.functional.cross_entropy(logits, labels, reduction="none")

    chunks = []
    for start in range(0, len(positions), 32):
        states, labels = hidden[start:start + 32], targets[start:start + 32]
        if torch.is_grad_enabled():
            chunks.append(checkpoint(logprob_chunk, states, labels, use_reentrant=False))
        else:
            chunks.append(logprob_chunk(states, labels))
    return torch.cat(chunks)
