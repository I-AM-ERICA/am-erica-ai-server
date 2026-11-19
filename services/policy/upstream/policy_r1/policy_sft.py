"""Offline, evidence-linked Korean search demonstrations. No network or training.

Approval is a recorded semantic review, NOT an automatic factuality claim.
Structural checks cannot decide whether a quoted passage entails an answer.
"""
import hashlib
import html
import json
import re
from urllib.parse import urlsplit, urlunsplit, parse_qs, urlencode

from .observations import observation_ids
from .official_pages import official_url
from .protocol import parse_action


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def content_digest(row):
    return fingerprint({k: v for k, v in row.items() if k != "review"})


def question_key(question):
    return re.sub(r"\W+", "", question.casefold())


def dated_question(row):
    # Date is input evidence for time-sensitive answers, not hidden review-only
    # metadata. Do not expose answer, expected outcome, split or teacher rationale.
    from datetime import date
    date.fromisoformat(row["as_of"])
    return f"기준일: {row['as_of']}\n{row['question']}"


def document_key(url):
    parts = urlsplit(url)
    # Bizinfo list/filter parameters do not identify a different notice. Preserve
    # all other URLs conservatively; do not conflate distinct notice IDs.
    if (parts.netloc.lower() == "www.bizinfo.go.kr"
            and parts.path == "/sii/siia/selectSIIA200Detail.do"):
        notice = parse_qs(parts.query).get("pblancId", [])
        if len(notice) == 1 and re.fullmatch(r"PBLN_[0-9]+", notice[0]):
            return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path,
                               urlencode({"pblancId": notice[0]}), ""))
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, parts.query, ""))


def validate_record(row, snapshots):
    for name in ("id", "question", "group_id", "agency", "as_of", "answer"):
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError(f"Missing {name}")
    if row.get("split") not in {"train", "val", "test"}:
        raise ValueError("Invalid split")
    review = row.get("review", {})
    if (review.get("status") != "approved" or not review.get("reviewer")
            or review.get("evidence_checked") is not True
            or review.get("coverage_checked") is not True
            or not review.get("notes") or review.get("content_sha256") != content_digest(row)):
        raise ValueError(f"{row['id']}: missing/stale semantic review")
    if row.get("outcome") not in {"supported", "insufficient"}:
        raise ValueError("Outcome must distinguish supported vs insufficient evidence")
    steps = row.get("steps", [])
    if not 1 <= len(steps) <= 2:
        raise ValueError("Require one or two real searches")
    documents = {}
    for i, step in enumerate(steps):
        query = step.get("query", "")
        if parse_action(f"<search>{query}</search>") != ("search", query):
            raise ValueError("Invalid search action")
        snapshot = snapshots.get(step.get("snapshot_key"))
        if (not snapshot or snapshot.get("error_code") or snapshot["query"] != query
                or snapshot["top_k"] != 3):
            raise ValueError("Search snapshot missing/mismatched/error; never train on provider failures")
        if i and not step.get("reason_for_next_search"):
            raise ValueError("Second search needs reviewed missing-evidence rationale")
        for hit in snapshot["hits"]:
            if hit.get("evidence_kind") != "official_page_excerpt" or not official_url(hit.get("url", "")):
                raise ValueError("Demonstration requires retrieved official HTML, not snippets")
            if not hit.get("snippet"):
                raise ValueError("Empty HTML evidence")
            documents.setdefault(hit["id"], []).append(hit)
    if parse_action(f"<answer>{row['answer']}</answer>") != ("answer", row["answer"]):
        raise ValueError("Invalid final answer")
    claims = row.get("claims", [])
    if row["outcome"] == "supported" and not claims:
        raise ValueError("Supported answer needs evidence-linked claims")
    if row["outcome"] == "insufficient" and not row.get("limitations"):
        raise ValueError("Abstention needs explicit unanswered requirements")
    for claim in claims:
        if not claim.get("text") or claim["text"] not in row["answer"] or not claim.get("evidence"):
            raise ValueError("Every annotated claim must occur in answer and cite evidence")
        for evidence in claim["evidence"]:
            quote = evidence.get("quote", "")
            field = evidence.get("field", "snippet")
            if field not in {"snippet", "title"}:
                raise ValueError("Evidence field must be snippet or title")
            if len(quote.strip()) < 4 or not any(quote in h.get(field, "") for h in documents.get(evidence.get("id"), [])):
                raise ValueError("Evidence quote not found in retrieved document")
    return documents


def validate_dataset(rows, snapshots, excluded_questions=(), excluded_urls=()):
    if not rows:
        raise ValueError("No reviewed demonstrations")
    ids, ownership = set(), {}
    excluded_q = {question_key(q) for q in excluded_questions}
    excluded_u = {document_key(u) for u in excluded_urls}
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate demonstration ID")
        ids.add(row["id"])
        documents = validate_record(row, snapshots)
        if question_key(row["question"]) in excluded_q:
            raise ValueError("Historical evaluation question cannot become SFT data")
        identities = [("group", row["group_id"]), ("question", question_key(row["question"]))]
        for versions in documents.values():
            for hit in versions:
                url = document_key(hit["url"])
                if row["split"] == "train" and url in excluded_u:
                    raise ValueError("Historical evaluation document cannot become training evidence")
                identities.extend([("url", url), ("body", fingerprint(hit["snippet"]))])
        for key in identities:
            if key in ownership and ownership[key] != row["split"]:
                raise ValueError(f"Cross-split leakage: {key[0]}")
            ownership[key] = row["split"]
    return {"records": len(rows), "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
            "note": "Exact question/URL/body/group checks do not replace semantic leakage review"}


def compile_record(row, snapshots, tokenizer, config, system_prompt):
    """Match collect_rollout tokens exactly; no gold inserted into observation.

    labels use the standard causal-LM convention (model shifts internally).
    Search and answer tokens receive loss; prompt and ALL observation tokens -100.
    """
    from .rollout import prompt_ids
    validate_record(row, snapshots)
    if (config.observation_strategy not in {"balanced_v1", "focused_v1"} or config.top_k != 3
            or len(row["steps"]) > config.max_searches):
        raise ValueError("SFT environment must match balanced_v1/top3/search limit")
    ids = prompt_ids(tokenizer, dated_question(row), config, system_prompt)
    mask, spans, visible = [False] * len(ids), [], {}

    def append(tokens, action, kind):
        start = len(ids)
        ids.extend(tokens)
        mask.extend([action] * len(tokens))
        spans.append({"type": kind, "start": start, "end": len(ids)})

    for step in row["steps"]:
        action = tokenizer.encode(f"<search>{step['query']}</search>", add_special_tokens=False)
        if len(action) > config.max_new_tokens:
            raise ValueError("Search action exceeds generation limit")
        append(action, True, "search")
        hits = snapshots[step["snapshot_key"]]["hits"]
        budget = min(config.max_observation_tokens, config.max_seq_length - len(ids) - config.max_new_tokens)
        obs = observation_ids(tokenizer, hits, budget, config.observation_strategy,
                              question=dated_question(row), query=step['query'])
        if obs is None:
            raise ValueError("No context space for observation")
        append(obs, False, "observation")
        text = tokenizer.decode(obs, skip_special_tokens=True)
        # Match evidence to its own visible document block, never to another hit.
        blocks = re.split(r"\[([\w:.-]{1,128})\] ", text)
        for index in range(1, len(blocks), 2):
            visible.setdefault(blocks[index], []).append(html.unescape(blocks[index + 1]))
    for claim in row.get("claims", []):
        for evidence in claim["evidence"]:
            if not any(evidence["quote"] in block for block in visible.get(evidence["id"], [])):
                raise ValueError(f"Claim evidence was truncated out of model-visible observation: {evidence['id']}")
    answer = tokenizer.encode(f"<answer>{row['answer']}</answer>", add_special_tokens=False)
    if len(answer) > config.max_new_tokens or len(ids) + len(answer) > config.max_seq_length:
        raise ValueError("Answer exceeds generation/context limit; never silently truncate labels")
    append(answer, True, "answer")
    return {"id": row["id"], "split": row["split"], "input_ids": ids, "action_mask": mask,
            "labels": [token if active else -100 for token, active in zip(ids, mask)],
            "spans": spans, "content_sha256": content_digest(row)}
