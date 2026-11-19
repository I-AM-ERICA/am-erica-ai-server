"""Deterministic, gold-free whole-section selection for bounded observations.

Inputs are only the retrieved text, question and model search query. Never infer
facts, rewrite sentences, execute HTML, or inspect labels/reviewer annotations.
"""
import math
import re

HEADINGS = re.compile(r'^(?:사업목적|사업개요|지원대상|지원내용|지원규모|지원절차|신청기간|'
                      r'신청[·ㆍ]접수|사업신청 방법|사업신청 사이트|접수마감 일정 안내|문의처|'
                      r'제외대상|지원제외|유의사항|주의사항|첨부파일)$')
CAUTION = re.compile(r'^[\s\ufeff]*(?:단[,，\s]|다만|예외|제외|주의|유의|※|\*)')
STOP = {'공식','자료','문서','출처','확인','확인해줘','알려줘','지원','사업','소상공인',
        '기준일','지역','업종','site','www','https','http','go','kr','or','co','com'}
INTENTS = (
    (('대상','자격'), ('지원대상',)),
    (('내용','방식','지급'), ('지원내용',)),
    (('기간','마감'), ('신청기간','접수마감 일정 안내')),
    (('절차','경로','어디','방법'), ('지원절차','신청·접수','사업신청 방법')),
    (('제외','제한','예외'), ('제외대상','지원제외','유의사항','주의사항')),
)


def terms(text):
    text = re.sub(r'https?://\S+|site:\S+', ' ', text.casefold())
    words = re.findall(r'[a-z0-9가-힣]{2,}', text)
    result = set()
    for word in words:
        word = re.sub(r'(?:에서는|으로|에서|에는|의|을|를|은|는|이|가|와|과)$', '', word)
        if len(word) >= 2 and word not in STOP:
            result.add(word)
    return result


def sections(text):
    # Preserve text verbatim inside each section. Merge attached caveats rather
    # than selecting a positive sentence while leaving its exception behind.
    lines = text.split('\n')
    result, current = [], []
    for line in lines:
        # A standalone plaintext table caption starts its own indivisible unit.
        # This keeps explanatory paragraphs usable without copying an entire
        # flattened numeric table, whose row/column relationships may be lost.
        caption = re.fullmatch(r'<[^<>\n]+>', line.strip())
        if not line.strip() or ((HEADINGS.fullmatch(line.strip()) or caption) and current):
            if current:
                result.append('\n'.join(current).strip())
                current = []
        if line.strip():
            current.append(line)
    if current:
        result.append('\n'.join(current).strip())
    grouped = []
    for block in result:
        if grouped and CAUTION.match(block):
            grouped[-1] += '\n\n' + block
        else:
            grouped.append(block)
    return grouped


def select_sections(text, question, query, encode, budget):
    """Return token IDs (encode must escape untrusted text) within exact budget."""
    if budget <= 0:
        return []
    complete = encode(text)
    if len(complete) <= budget:
        return complete
    units = sections(text)
    qt, st = terms(question), terms(query)
    all_terms = qt | st
    scores = []
    for index, unit in enumerate(units):
        lower = unit.casefold()
        heading = lower.split('\n')[0]
        score = 0.0
        for term in all_terms:
            if term in lower:
                frequency = sum(term in u.casefold() for u in units)
                weight = 2.0 if term in st else 1.0
                score += weight * (1 + math.log((len(units)+1)/(frequency+1)))
                if term in heading:
                    score += weight
        # Flattened tables repeat topic words in captions/column labels. Unless
        # the user asks for numeric/table detail, prefer explanatory prose over
        # that repetition. No answer terms or source-specific rules are used.
        if re.fullmatch(r'<[^<>\n]+>', unit.split('\n')[0].strip()):
            if not any(t in question for t in ('금액','비율','등급','얼마','표로','한도')):
                score *= 0.2
        # Intent headings are generic document structure, not policy answers.
        # A question about eligibility/payment should not be displaced by a
        # contact/address section that happens to repeat the program's name.
        for cues, headings in INTENTS:
            if any(cue in question for cue in cues) and heading.strip() in headings:
                score += 12
        scores.append((score, index))
    # One generic omission marker also prevents disjoint excerpts from looking
    # like an uninterrupted paragraph. Restore source order after selection.
    marker = '\n…\n'
    selected = []
    def render(indices):
        return marker + marker.join(units[i] for i in sorted(indices)) + marker
    for _, index in sorted(scores, key=lambda item: (-item[0], item[1])):
        candidate = selected + [index]
        if len(encode(render(candidate))) <= budget:
            selected = candidate
    if selected:
        return encode(render(selected))
    # Never clip a long sentence/table halfway through a condition or negation.
    unavailable = encode('본문 문단이 길어 이 관측에서 생략됨')
    return unavailable if len(unavailable) <= budget else []
