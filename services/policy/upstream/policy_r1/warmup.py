# 역할: 학습 split의 실제 검색 결과로 검색/답변 형식 SFT 예제를 만듭니다. 평가 정답은 사용하지 않습니다.
from collections import Counter
import random
import re
import warnings

from .protocol import normalize_answer, parse_action
from .rollout import Trajectory, observation_ids, prompt_ids


def build_protocol_examples(tokenizer, retriever, train_rows, config):
    """Teacher-forced demonstrations, explicitly separate from on-policy PPO rollouts."""
    if config.max_searches < 1:
        raise ValueError("검색 형식 warmup에는 max_searches >= 1이 필요합니다.")
    examples, skipped = [], Counter()
    rows = train_rows
    if config.warmup_max_candidates:
        rows = random.Random(config.seed).sample(train_rows, min(len(train_rows), config.warmup_max_candidates))
    for index, row in enumerate(rows, 1):
        try:
            examples.append(_build_example(tokenizer, retriever, row, config))
        except ValueError as error:
            if not config.warmup_skip_unusable:
                raise
            skipped[str(error).split(": ", 1)[-1]] += 1
        if config.warmup_max_examples and len(examples) >= config.warmup_max_examples:
            break
        if index % 50 == 0:
            print(f"Warmup candidates={index}, usable={len(examples)}, skipped={sum(skipped.values())}", flush=True)
    if skipped:
        warnings.warn(f"Warmup uses only retrieved, visible-answer examples; skipped={dict(skipped)}")
    if not examples:
        raise ValueError("warmup에 사용할 검색 근거가 있는 예제가 없습니다. 검색 결과/observation 길이를 확인하세요.")
    print(f"Warmup examples={len(examples)} (train only)", flush=True)
    return examples


def _build_example(tokenizer, retriever, row, config):
    ids = prompt_ids(tokenizer, row["question"], config)
    mask = [False] * len(ids)
    query = row["question"]
    search_text = f"<search>{query}</search>"
    if parse_action(search_text) != ("search", query):
        raise ValueError(f"{row['id']}: 질문에 protocol 태그가 들어 있어 warmup 질의로 사용할 수 없습니다.")
    search_ids = tokenizer.encode(search_text, add_special_tokens=False)
    hits = retriever.search(query, config.top_k)
    evidence = row.get("evidence_ids", [])
    if (config.warmup_require_evidence and not evidence) or (evidence and not set(evidence).intersection(h["id"] for h in hits)):
        raise ValueError(f"{row['id']}: warmup에는 검색된 근거 문서를 가리키는 evidence_ids가 필요합니다.")
    ids.extend(search_ids)
    mask.extend([True] * len(search_ids))
    # Match the online environment's observation budget, including room for the next action.
    budget = min(config.max_observation_tokens,
                 config.max_seq_length - len(ids) - config.max_new_tokens)
    obs = observation_ids(tokenizer, hits, budget)
    if obs is None:
        raise ValueError(f"{row['id']}: 검색 결과를 넣을 context 공간이 없습니다.")
    observation = tokenizer.decode(obs, skip_special_tokens=True)
    # Document identifiers are metadata, not answer evidence. English answers match whole tokens.
    visible_text = re.sub(r"(?m)^\[[^\]\n]*\]\s*", "", observation)
    normalized = normalize_answer(visible_text, config.answer_normalization)
    def visible(answer):
        target = normalize_answer(answer, config.answer_normalization)
        if config.answer_normalization == "search_r1":
            return bool(target) and f" {target} " in f" {normalized} "
        return bool(target) and target in normalized
    answer = next((a for a in row["answers"] if visible(a)), None)
    if answer is None:
        raise ValueError(f"{row['id']}: 잘린 검색 결과에서 정답을 확인할 수 없습니다. 검색 결과와 observation 길이를 확인하세요.")
    answer_text = f"<answer>{answer}</answer>"
    if parse_action(answer_text) != ("answer", answer):
        raise ValueError(f"{row['id']}: warmup 정답에 protocol 태그가 들어 있습니다.")
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if max(len(search_ids), len(answer_ids)) > config.max_new_tokens:
        raise ValueError(f"{row['id']}: warmup action이 max_new_tokens보다 깁니다.")
    ids.extend(obs)
    mask.extend([False] * len(obs))
    ids.extend(answer_ids)
    mask.extend([True] * len(answer_ids))
    if len(ids) > config.max_seq_length:
        raise ValueError(f"{row['id']}: warmup trajectory가 max_seq_length를 초과합니다.")
    return Trajectory(ids, mask, answer, 1, "answer", [
        {"type": "search", "text": search_text},
        {"type": "observation", "document_ids": [h["id"] for h in hits], "text": observation},
        {"type": "answer", "text": answer_text},
    ])
