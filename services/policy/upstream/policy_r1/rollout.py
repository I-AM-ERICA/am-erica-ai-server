# 역할: Actor 생성 → 문서 검색 → 추가 생성 루프와 PPO용 action-token mask를 만듭니다.
from dataclasses import dataclass
import html

import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from .protocol import SYSTEM_PROMPT, ENGLISH_SYSTEM_PROMPT, parse_action
from .observations import observation_ids


MODEL_STOP_STRINGS = ("<|endofturn|>", "<|stop|>")


class StopAtAction(StoppingCriteria):
    def __init__(self, tokenizer, start):
        self.tokenizer, self.start = tokenizer, start

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.start:], skip_special_tokens=True)
        raw = self.tokenizer.decode(input_ids[0, self.start:], skip_special_tokens=False)
        return "</search>" in text or "</answer>" in text or any(marker in raw for marker in MODEL_STOP_STRINGS)


class RolloutCancelled(RuntimeError):
    pass


class CheckCancellation(StoppingCriteria):
    def __init__(self, should_stop):
        self.should_stop = should_stop

    def __call__(self, input_ids, scores, **kwargs):
        if self.should_stop():
            raise RolloutCancelled("Request cancelled or deadline exceeded")
        return False


@dataclass
class Trajectory:
    input_ids: list
    action_mask: list
    answer: str
    searches: int
    finish_reason: str
    events: list

    def tensors(self, device):
        ids = torch.tensor([self.input_ids], dtype=torch.long, device=device)
        positions = torch.tensor([i for i, active in enumerate(self.action_mask) if active],
                                 dtype=torch.long, device=device)
        if not len(positions) or int(positions.min()) < 1:
            raise ValueError("유효한 action token이 없습니다.")
        return ids, positions


def prompt_ids(tokenizer, question, config, system_prompt=None):
    messages = [{"role": "tool_list", "content": ""},
                {"role": "system", "content": system_prompt if system_prompt is not None else
                 (ENGLISH_SYSTEM_PROMPT if config.prompt_language == "en" else SYSTEM_PROMPT)},
                {"role": "user", "content": question}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if len(ids) > config.max_prompt_tokens:
        raise ValueError(f"질문 prompt가 {len(ids)} tokens입니다. max_prompt_tokens를 늘리거나 질문을 줄이세요.")
    return ids


@torch.no_grad()
def collect_rollout(actor, tokenizer, retriever, question, config, sample=True, *, system_prompt=None, should_stop=None,
                    initial_hits=None):
    actor.eval()
    ids = prompt_ids(tokenizer, question, config, system_prompt)
    check_cancelled = CheckCancellation(should_stop or (lambda: False))
    mask = [False] * len(ids)
    events, answer, searches = [], "", 0
    if initial_hits:
        check_cancelled(None, None)
        budget = min(config.max_observation_tokens, config.max_seq_length - len(ids) - config.max_new_tokens)
        obs_ids = observation_ids(tokenizer, initial_hits, budget, config.observation_strategy, question=question)
        if obs_ids is None:
            raise ValueError("No room for source pages in the model context")
        ids.extend(obs_ids)
        mask.extend([False] * len(obs_ids))
        events.append({"type": "observation", "document_ids": [h["id"] for h in initial_hits],
                       "text": tokenizer.decode(obs_ids)})
    reason = "turn_limit"
    eos = actor.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = list(eos) if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
    # HyperCLOVA's official generation example stops at both turn-end markers.
    # Include atomic special-token IDs; StopAtAction also handles multi-token spellings.
    vocabulary = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for marker in MODEL_STOP_STRINGS:
        if marker in vocabulary and vocabulary[marker] not in eos_ids:
            eos_ids.append(vocabulary[marker])
    eos = eos_ids or None
    # 모델 기본 generation_config의 top-k/repetition 설정을 상속하지 않습니다.
    generation = GenerationConfig(
        do_sample=sample,
        # Greedy evaluation does not use sampling-only flags (and must not warn about top_k).
        **({"temperature": 1.0, "top_k": 0, "top_p": 1.0} if sample else {}),
        repetition_penalty=1.0, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, use_cache=True,
    )
    for _ in range(config.max_searches + 1):
        check_cancelled(None, None)
        remaining = config.max_seq_length - len(ids)
        if remaining <= 0:
            reason = "context_limit"
            break
        tensor = torch.tensor([ids], device=next(actor.parameters()).device)
        token_budget = min(config.max_new_tokens, remaining)
        output = actor.generate(
            input_ids=tensor, attention_mask=torch.ones_like(tensor),
            generation_config=generation,
            max_new_tokens=token_budget,
            stopping_criteria=StoppingCriteriaList([check_cancelled, StopAtAction(tokenizer, len(ids))]),
        )
        check_cancelled(None, None)
        new_ids = output[0, len(ids):].tolist()
        # 생성 token id를 decode→encode하지 않아 old/new log-prob의 정렬을 보존합니다.
        ids.extend(new_ids)
        mask.extend([True] * len(new_ids))
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        action, content = parse_action(text)
        hit_limit = len(new_ids) >= token_budget
        raw_text = tokenizer.decode(new_ids, skip_special_tokens=False)
        model_turn_end = any(marker in raw_text for marker in MODEL_STOP_STRINGS)
        events.append({"type": action, "text": text, "generated_tokens": len(new_ids),
                       "hit_token_limit": hit_limit, "model_turn_end": model_turn_end})
        if action == "answer":
            answer, reason = content, "answer"
            break
        if action != "search":
            if model_turn_end:
                reason = "model_turn_end"
            elif hit_limit:
                reason = "context_limit" if remaining <= config.max_new_tokens else "generation_limit"
            else:
                reason = "invalid_action"
            break
        if searches >= config.max_searches:
            reason = "search_limit"
            break
        searches += 1
        hits = retriever.search(content, config.top_k)
        check_cancelled(None, None)
        # Escape untrusted document text and mask every observation token out of the loss.
        budget = min(config.max_observation_tokens,
                     config.max_seq_length - len(ids) - config.max_new_tokens)
        obs_ids = observation_ids(tokenizer, hits, budget, config.observation_strategy, question=question, query=content)
        if obs_ids is None:
            reason = "context_limit"
            break
        ids.extend(obs_ids)
        mask.extend([False] * len(obs_ids))
        events.append({"type": "observation", "document_ids": [h["id"] for h in hits],
                       "text": tokenizer.decode(obs_ids)})
    return Trajectory(ids, mask, answer, searches, reason, events)
