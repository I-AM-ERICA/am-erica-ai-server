"""Conservative serving-only cleanup of echoed context and request phrasing."""
import re


def fallback_search_query(question):
    """Recover a missing model query from the user's words, preserving constraints."""
    if question.startswith("다음은 같은 사용자의 이전 대화입니다.") and "\n현재 사용자 질문: " in question:
        question = question.rsplit("\n현재 사용자 질문: ", 1)[1]
    original = " ".join(question.split())[:1000]
    text = re.sub(r"(?:가|이)\s*받을\s*수\s*있는", " ", original)
    text = re.sub(r"(?:본문의?\s*)?지원\s*내용과\s*신청\s*조건", " ", text)
    text = re.sub(r"(?:을|를)?\s*(?:찾아서|찾아\s*줘|찾아\s*주라|요약해\s*줘|요약해\s*주세요|알려\s*줘|알려\s*주세요)[.!?。]*", " ", text)
    text = re.sub(r"(?<!\S)(?:나는|저는|현재|본문의|본문)(?=\s|$)", " ", text)
    text = " ".join(text.replace(",", " ").split()).strip(" .!?。")
    return text or original


def requests_official_sources(question):
    # The backend prepends prior conversation. Apply this constraint to the latest
    # question, not to an old mention that might no longer be relevant.
    if question.startswith("다음은 같은 사용자의 이전 대화입니다.") and "\n현재 사용자 질문: " in question:
        question = question.rsplit("\n현재 사용자 질문: ", 1)[1]
    return bool(re.search(r"(?<!비)공식\s*(?:자료|문서|출처|공고|사이트|홈페이지)", question))


def clean_search_query(query, context_prefix=""):
    text = " ".join(query.split())
    # Match the exact context values at the beginning, allowing the model to
    # replace line breaks with punctuation. Never normalize slashes globally:
    # a date, URL or an explicit constraint in the query must stay intact.
    if context_prefix.strip():
        fields = [r"\s*".join(re.escape(word) for word in line.split())
                  for line in context_prefix.splitlines() if line.strip()]
        separator = r"(?:\s*[/|;,]\s*|\s+)"
        prefix = separator.join(fields)
        # Require a boundary after the last value (음식점 must not match 음식점업).
        text = re.sub(r"^" + prefix + r"(?:" + separator + r"|$)", "", text, count=1)
    # Strip a trailing conversational instruction, not content in the middle.
    text = re.sub(
        r"(?:을|를)?\s*(?:공식\s*(?:자료|문서|출처)(?:로|에서)?\s*)?"
        r"(?:확인해\s*줘|알려\s*줘|찾아\s*줘|검색해\s*줘|확인해\s*주세요|알려\s*주세요|찾아\s*주세요)"
        r"[.!?。]*$", "", text).strip()
    text = re.sub(r"(정책자금|지원사업|지원제도|정책)의(?=\s|$)", r"\1", text)
    return text


def broader_search_query(query):
    # Broaden only an eligibility-information phrase. Preserve entities, dates,
    # locations, exclusions, amounts and search operators; do not take first N words.
    broader = re.sub(r"(?<!\S)공통\s+지원\s+자격(?=\s|$)", "지원", query)
    broader = " ".join(broader.split())
    return broader if broader != query and broader else None
