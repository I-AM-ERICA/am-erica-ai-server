"""Bundled OFL fonts; IDs are stable across saved designs, UI and CLOVA."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TypographyId = Literal["auto", "sans", "serif", "bold", "gowun_batang", "gowun_dodum", "jua", "dohyeon", "black_han_sans", "nanum_pen"]
ROOT = Path(__file__).parent / "fonts"


@dataclass(frozen=True)
class FontChoice:
    id: str
    name: str
    description: str
    path: str = ""


FONT_CHOICES = [
    FontChoice("auto", "자동 추천", "상품과 구도에 어울리는 서체"),
    FontChoice("sans", "나눔고딕", "담백하고 또렷한 기본 고딕", "nanumgothic/NanumGothic-Regular.ttf"),
    FontChoice("bold", "나눔고딕 Bold", "선명하고 힘 있는 제목", "nanumgothic/NanumGothic-Bold.ttf"),
    FontChoice("serif", "나눔명조", "차분한 프리미엄·공방·브랜드", "NanumMyeongjo-Regular.ttf"),
    FontChoice("gowun_batang", "고운바탕", "부드러운 명조·선물·뷰티", "gowunbatang/GowunBatang-Regular.ttf"),
    FontChoice("gowun_dodum", "고운돋움", "따뜻한 고딕·카페·생활용품", "gowundodum/GowunDodum-Regular.ttf"),
    FontChoice("jua", "배민 주아", "둥글고 친근한 제목·간식·소품", "jua/Jua-Regular.ttf"),
    FontChoice("dohyeon", "배민 도현", "단단한 제목·패션·전자제품", "dohyeon/DoHyeon-Regular.ttf"),
    FontChoice("black_han_sans", "검은고딕", "굵고 대담한 캠페인·행사", "blackhansans/BlackHanSans-Regular.ttf"),
    FontChoice("nanum_pen", "나눔손글씨 펜", "손으로 쓴 듯한 공방·꽃·디저트", "nanumpenscript/NanumPenScript-Regular.ttf"),
]
FONT_BY_ID = {font.id: font for font in FONT_CHOICES}


def public_fonts():
    return [{"id": f.id, "name": f.name, "description": f.description} for f in FONT_CHOICES]
