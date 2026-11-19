"""Apply unambiguous design controls without leaving enum selection to the LLM."""
import re
from .templates import TEMPLATES
from .font_catalog import FONT_CHOICES

PALETTE_LABELS = {
    "cream": "크림·브라운: 따뜻한 저채도, 담백한 여백",
    "coral": "코랄 레드·크림·레몬: 선명한 대비, 행사 강조",
    "sage": "연한 세이지·짙은 녹색·오렌지: 산뜻하고 부드러운 대비",
    "cobalt": "아이스 블루·네이비·코발트: 깨끗하고 또렷한 인상",
    "ink": "차콜·아이보리·골드: 차분하고 깊은 대비",
    "peach": "살구·크림·테라코타: 포근한 온기",
    "burgundy": "아이보리·버건디: 대담한 글자와 빈티지한 온기",
    "forest": "딥 그린·아이보리·골드: 깊고 고요한 인상",
    "sky": "선명한 블루·화이트: 맑고 활기찬 대비",
    "tangerine": "번트 오렌지·크림: 풍성하고 따뜻한 대비",
    "scarlet": "오프화이트·스칼렛 레드: 여백과 큰 글자가 돋보이는 대비",
}


def explicit_edits(instruction):
    suffix = r"(?:\s*(?:바꿔\s*줘|바꿔\s*주세요|변경해\s*줘|변경해\s*주세요|해\s*줘|해\s*주세요))?"
    colors = {"초록색": "sage", "초록": "sage", "파란색": "cobalt", "파랑": "cobalt",
              "분홍색": "peach", "분홍": "peach", "빨간색": "coral", "빨강": "coral",
              "크림색": "cream", "어두운 색": "ink", "버건디": "burgundy", "짙은 초록색": "forest"}
    fonts = {"명조체": "serif", "명조": "serif", "고딕체": "sans", "고딕": "sans", "굵은 고딕": "bold"}
    fonts.update({name: f.id for f in FONT_CHOICES for name in (f.id, f.name)})
    fonts.update({"주아체": "jua", "주아": "jua", "도현체": "dohyeon", "도현": "dohyeon", "검은 고딕": "black_han_sans", "나눔펜": "nanum_pen"})
    templates = {name: t for t in TEMPLATES for name in (t.id, t.name)}
    changes = {}
    clauses = re.split(r"\s*(?:,|그리고|바꾸고|변경하고)\s*", instruction.strip().rstrip(".!"))
    for clause in clauses:
        footer = re.sub(r"[\s\"'“”‘’]", "", clause)
        target = r"(?:(?:오른쪽아래|우측하단)의?)?(?:하단안내(?:문구)?|지역미설정|지역안내|방문·?문의안내)(?:을|를|는|은)?"
        if re.fullmatch(target + r"(?:숨기기|숨겨(?:줘|주세요)?|지워(?:줘|주세요)?|삭제(?:해줘|해주세요|하기)?|없애(?:줘|주세요)?|빼(?:줘|주세요)?)", footer):
            changes["hide_footer"] = True
            continue
        if re.fullmatch(target + r"(?:다시)?(?:표시(?:해줘|해주세요|하기)?|보여(?:줘|주세요)?|보이게해(?:줘|주세요))", footer):
            changes["hide_footer"] = False
            continue
        color = re.fullmatch(r"(?:배경[을은]?\s*)?("+"|".join(colors)+r")(?:으로|로)"+suffix, clause)
        font = re.fullmatch(r"(?:제목(?:\s*서체)?[을은를]?\s*)?("+"|".join(map(re.escape, fonts))+r")(?:으로|로)"+suffix, clause)
        template = re.fullmatch(r"("+"|".join(map(re.escape, templates))+r")\s*템플릿으로"+suffix, clause)
        if color:
            changes["palette"] = colors[color[1]]
        elif font:
            changes["typography"] = fonts[font[1]]
        elif template:
            selected = templates[template[1]]
            changes.update(template_id=selected.id, palette=selected.palette, typography="auto")
        elif re.fullmatch(r"가격[을은]?\s*(?:더\s*)?강조"+suffix, clause):
            changes["emphasis"] = "price"
        elif clause in ("사진 전체 보기", "사진 전체가 보이게 해줘", "사진 전체"):
            changes["image_fit"] = "contain"
        elif clause in ("사진 꽉 채워줘", "사진 꽉 채우기"):
            changes["image_fit"] = "cover"
        else:
            # Complex instructions, negations and fact edits go through the full designer.
            return None
    return changes or None
