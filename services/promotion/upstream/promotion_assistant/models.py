from __future__ import annotations

import base64
from .copy_guard import unsupported_claims
from .font_catalog import TypographyId
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

TemplateId = Literal["editorial", "bold_sale", "soft_frame", "split", "night", "market", "type_stack", "magazine", "spotlight", "diagonal", "floating_type", "big_focus"]
PaletteId = Literal["cream", "coral", "sage", "cobalt", "ink", "peach", "burgundy", "forest", "sky", "tangerine", "scarlet"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Store(Model):
    store_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=40)
    category: str = Field(default="", max_length=60)
    region: str = Field(default="", max_length=100)
    address: str = Field(default="", max_length=150)


class ImageInput(Model):
    base64: str = Field(min_length=1, max_length=7_000_000)
    media_type: Literal["image/png", "image/jpeg", "image/webp"]

    def decode(self) -> bytes:
        try:
            data = base64.b64decode(self.base64, validate=True)
        except (ValueError, base64.binascii.Error):
            raise ValueError("이미지의 base64 형식이 올바르지 않습니다.") from None
        if not 0 < len(data) <= 5 * 1024 * 1024:
            raise ValueError("이미지는 5MB 이하여야 합니다.")
        return data


class Brief(Model):
    store: Store
    product: str = Field(min_length=1, max_length=100)
    message: str = Field(default="", max_length=500)
    tone: str = Field(default="따뜻하고 친근하게", max_length=80)
    channel: str = Field(default="instagram", max_length=30)
    price: int | None = Field(default=None, ge=0, le=999_999_999)
    original_price: int | None = Field(default=None, ge=1, le=999_999_999)
    period: str = Field(default="", max_length=60)
    contact: str = Field(default="", max_length=100)
    hide_footer: bool = False
    background_mode: Literal["auto", "remove", "keep"] = "auto"
    product_image: ImageInput | None = None
    logo: ImageInput | None = None
    style_reference: ImageInput | None = None

    @model_validator(mode="after")
    def prices(self):
        if self.original_price is not None and (self.price is None or self.original_price <= self.price):
            raise ValueError("정가는 판매가보다 커야 하며 판매가도 입력해야 합니다.")
        return self

    def facts(self):
        return self.model_dump(exclude={"product_image", "logo", "style_reference"})


class Design(Model):
    template_id: TemplateId
    palette: PaletteId
    headline: str = Field(min_length=1, max_length=40)
    subhead: str = Field(default="", max_length=90)
    cta: str = Field(default="매장에서 만나보세요", max_length=24)
    emphasis: Literal["product", "price"] = "product"
    image_fit: Literal["contain", "cover"] = "contain"
    typography: TypographyId = "auto"
    hide_footer: bool = False

    @field_validator("headline", "subhead", "cta")
    @classmethod
    def supported_text(cls, value):
        # Font coverage is deliberately Korean/Latin; emoji are not rasterised as tofu.
        if any(ord(c) > 0xFFFF or ord(c) < 32 and c not in "\n\t" for c in value):
            raise ValueError("문구에는 한글·영문·기본 기호만 사용할 수 있습니다.")
        return value


class DesignBatch(Model):
    designs: list[Design] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def distinct(self):
        if len({d.template_id for d in self.designs}) != 3:
            raise ValueError("세 시안은 서로 다른 템플릿이어야 합니다.")
        return self


class Revision(Model):
    store_id: str = Field(min_length=1, max_length=100)
    instruction: str = Field(min_length=1, max_length=500)
    expected_version: int = Field(ge=1)


def validate_copy(design: Design, brief: Brief):
    """Numeric claims may only repeat numeric facts supplied by the owner."""
    if unsupported_claims("\n".join([design.headline, design.subhead, design.cta]), brief):
        raise ValueError("입력 정보에 없는 숫자가 문구에 포함됐습니다. 다시 생성해 주세요.")
    if design.emphasis == "price" and brief.price is None:
        design = design.model_copy(update={"emphasis": "product"})
    return design
