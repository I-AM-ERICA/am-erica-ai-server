"""Serving-only source continuity and bounded, explicitly partial page evidence."""
import re

from .official_pages import public_page_url


def previous_summary_urls(question):
    if not question.startswith("다음은 같은 사용자의 이전 대화입니다."):
        return []
    history, separator, current = question.rpartition("\n현재 사용자 질문: ")
    if not separator or not re.search(r"요약|정리|그\s*(?:내용|정책|자료)|링크|어떤\s*내용|자세히", current):
        return []
    urls = re.findall(r"^이전 답변 출처 URL: (https://\S+)\s*$", history, re.MULTILINE)
    return list(dict.fromkeys(url for url in urls[-3:] if public_page_url(url)))


def page_observations(hits, tokenizer, budget):
    """Share the observation budget so the first long page cannot hide other sources."""
    pages = [hit for hit in hits if hit.get("evidence_kind") in {"official_page_excerpt", "page_excerpt"}]
    result = []
    for hit in pages:
        header = f"제목: {hit['title'][:100]}\n읽은 HTML 본문 일부(다른 조건·첨부파일 미확인): "
        overhead = len(tokenizer.encode(f"[{hit['id']}] {header}", add_special_tokens=False))
        allowance = (budget - 48) // len(pages) - overhead - 16
        if allowance < 32:
            continue
        body = tokenizer.decode(tokenizer.encode(hit["snippet"], add_special_tokens=False)[:allowance],
                                skip_special_tokens=True)
        copied = dict(hit)
        copied["snippet"] = body
        copied["contents"] = header + body
        result.append(copied)
    return result


def link_only_answer(answer):
    text = answer.strip()
    return bool(re.fullmatch(r"https?://\S+", text) or (
        len(text) < 110 and re.search(
            r"(?:정책|지원|사업)\s*(?:정보|안내)(?:입니다|예요)|(?:링크|공고|홈페이지)를?\s*확인(?:하세요|해\s*주세요)", text)))
