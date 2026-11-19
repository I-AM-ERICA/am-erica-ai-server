"""Select a short, relevant HTML section verbatim; never synthesize policy facts."""
import re


def focus_official_excerpt(hit, query):
    if hit.get("evidence_kind") not in {"official_page_excerpt", "page_excerpt"}:
        return hit
    original = hit.get("snippet", "")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", original) if block.strip()]
    if len(blocks) < 2:
        return hit
    terms = list(dict.fromkeys(re.findall(r"[a-z0-9가-힣]{2,}", query.lower())))
    terms = [word for word in terms if word not in {"공식", "자료", "문서", "출처", "확인해줘"}]
    candidates = []
    for index, block in enumerate(blocks):
        if not 40 <= len(block) <= 1200:
            continue
        text = " ".join(block.lower().split())
        heading = block.splitlines()[0].lower()
        matches = [word for word in terms if word in text]
        if len(matches) < 2 and not any(len(word) >= 4 and word in heading for word in matches):
            continue
        score = sum(min(len(word), 6) + (12 if word in heading else 0) for word in matches)
        # Adjacent query phrases distinguish an exact section heading from generic mentions.
        score += sum(15 for a, b in zip(terms, terms[1:]) if f"{a} {b}" in heading)
        candidates.append((score, -len(block), -index, index))
    if not candidates:
        return hit
    index = max(candidates)[3]
    selected = [blocks[index]]
    # Keep an immediately following exclusions/caution section with its parent.
    if index + 1 < len(blocks) and re.match(r"^(제외|예외|주의|유의|제한|단[,，])", blocks[index + 1]):
        if sum(map(len, selected)) + len(blocks[index + 1]) <= 1600:
            selected.append(blocks[index + 1])
        else:
            return hit  # do not silently drop a long adjoining caveat
    excerpt = "\n\n".join(selected)
    if len(excerpt) >= len(original):
        return hit
    focused = dict(hit)
    focused["snippet"] = excerpt
    focused["evidence_focus"] = {"original_chars": len(original), "selected_chars": len(excerpt),
                                 "section_heading": blocks[index].splitlines()[0]}
    label = "공식 HTML" if hit.get("evidence_kind") == "official_page_excerpt" else "HTML"
    focused["contents"] = (f"제목: {hit['title']}\nURL: {hit['url']}\n"
                           f"질문 관련 {label} 본문 발췌(일부; 다른 조건·첨부파일 미확인): {excerpt}")
    return focused
