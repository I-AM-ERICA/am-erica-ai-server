"""Versioned observation rendering shared by live rollouts and offline SFT.

concat preserves historical Wiki experiments. balanced_v1 gives each returned
document a complete ID and a fair share of the *actual remaining* token budget.
It performs no query rewriting, source insertion, or relevance judgement.
"""
import html
import re
from .paragraph_selection import select_sections


def observation_ids(tokenizer, hits, budget, strategy="concat", *, question="", query=""):
    if strategy not in {"concat", "balanced_v1", "focused_v1"}:
        raise ValueError("Unknown observation strategy")
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix, suffix = encode("\n<information>\n"), encode("\n</information>\n")
    available = budget - len(prefix) - len(suffix)
    if available <= 0:
        return None
    if strategy == "concat" or not hits:
        body = "\n".join(f"[{h['id']}] {h['contents']}" for h in hits) or "검색 결과 없음"
        return prefix + encode(html.escape(body))[:available] + suffix
    blocks, seen, focused = [], set(), []
    safe_encode = lambda text: encode(html.escape(text).replace("[", "&#91;").replace("]", "&#93;"))
    for hit in hits:
        key = hit["id"]
        if not isinstance(key, str) or not re.fullmatch(r"[\w:.-]{1,128}", key) or key in seen:
            raise ValueError("Observation IDs must be unique, bounded identifiers")
        seen.add(key)
        # URL-to-ID mapping stays in the result trace / source list. A long URL
        # must not consume the entire allowance before the first body sentence.
        if "snippet" in hit and hit.get("evidence_kind") in {"official_page_excerpt", "page_excerpt", "search_snippet"}:
            label = {"official_page_excerpt": "공식 HTML 본문 일부; 첨부파일 미확인",
                     "page_excerpt": "HTML 본문 일부; 공식 여부 미확인",
                     "search_snippet": "검색 요약; 원문 미확인"}[hit["evidence_kind"]]
            meta = f"제목: {hit.get('title', '')[:100]}\n{label}\n"
            if strategy == "focused_v1":
                meta = f"제목: {hit.get('title', '')[:100]}\n{label}; 질문 관련 발췌, 다른 조건 생략 가능\n"
            source = hit['snippet']
            contents = meta + source
        else:
            meta, source = "", hit["contents"]
            contents = hit["contents"]
        header = encode(html.escape(f"[{key}] "))
        # Document-provided fake [IDs] must not masquerade as source boundaries.
        body = encode(html.escape(contents).replace("[", "&#91;").replace("]", "&#93;"))
        blocks.append((header, body))
        focused.append((meta, source))
    newline = encode("\n")
    # Never emit half a document identifier. Under an extremely small context,
    # omit entire trailing documents, and keep at least one body token per ID.
    while blocks and sum(len(h) + min(1, len(b)) for h, b in blocks) + len(newline) * (len(blocks) - 1) > available:
        blocks.pop()
    if not blocks:
        return None
    remaining = available - sum(len(h) for h, _ in blocks) - len(newline) * (len(blocks) - 1)
    allocations = [0] * len(blocks)
    # Water filling: short documents return unused quota to longer documents.
    active = list(range(len(blocks)))
    while remaining and active:
        share = max(1, remaining // len(active))
        for i in active:
            take = min(share, len(blocks[i][1]) - allocations[i], remaining)
            allocations[i] += take
            remaining -= take
        active = [i for i in active if allocations[i] < len(blocks[i][1])]
    result = list(prefix)
    for i, ((header, body), count) in enumerate(zip(blocks, allocations)):
        if i:
            result.extend(newline)
        result.extend(header)
        if strategy == "focused_v1":
            meta, source = focused[i]
            meta_ids = safe_encode(meta)
            if len(meta_ids) <= count:
                result.extend(meta_ids)
                result.extend(select_sections(source, question, query, safe_encode, count-len(meta_ids)))
            # Too little room for metadata: emit the complete ID only.
        else:
            result.extend(body[:count])
    return result + suffix
