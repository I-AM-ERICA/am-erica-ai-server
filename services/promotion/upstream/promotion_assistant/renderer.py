"""CPU-only 1080x1350 raster renderer with measured Korean text layout."""
from __future__ import annotations
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import os
import warnings
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageChops
from .product_image import has_transparency
from .templates import PALETTES, TEMPLATES, default_font
import math
from .typography import wrap_text
from .font_catalog import FONT_BY_ID, ROOT as FONT_ROOT

WIDTH, HEIGHT = 1080, 1350
Image.MAX_IMAGE_PIXELS = 20_000_000


def normalize_image(source, *, max_edge=2000) -> bytes | None:
    if source is None:
        return None
    data = source.decode()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in ("PNG", "JPEG", "WEBP"):
                    raise ValueError("PNG, JPEG, WEBP만 지원합니다.")
                actual = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[image.format]
                if source.media_type != actual:
                    raise ValueError("이미지 형식과 media_type이 일치하지 않습니다.")
                if image.width * image.height > 20_000_000 or min(image.size) < 4:
                    raise ValueError("이미지 해상도를 확인해 주세요. 최대 2천만 화소입니다.")
                image = ImageOps.exif_transpose(image).convert("RGBA")
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                # Preserve alpha, remove EXIF/profile/metadata; never fetch caller URLs.
                clean = Image.new("RGBA", image.size)
                clean.paste(image)
                buffer = BytesIO()
                clean.save(buffer, "PNG")
                return buffer.getvalue()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, OSError):
        raise ValueError("이미지를 읽을 수 없거나 해상도가 너무 큽니다.") from None


def vision_image(data):
    """Pad extreme ratios to meet HCX-005's 1:5..5:1 input rule."""
    im = Image.open(BytesIO(data)).convert("RGB")
    w, h = im.size
    canvas = Image.new("RGB", (max(w, (h + 4) // 5), max(h, (w + 4) // 5)), "#F5F0E7")
    canvas.paste(im, ((canvas.width-w)//2, (canvas.height-h)//2))
    output = BytesIO()
    canvas.save(output, "PNG")
    return output.getvalue()


class LayoutError(ValueError):
    pass


def find_font(custom="", bold=False):
    candidates = [custom] if custom else []
    candidates += [
        str(FONT_ROOT / ("nanumgothic/NanumGothic-Bold.ttf" if bold else "nanumgothic/NanumGothic-Regular.ttf")),
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("한글 폰트가 없습니다. POSTER_FONT_PATH / POSTER_BOLD_FONT_PATH를 설정해 주세요.")


class Renderer:
    def __init__(self, font_path="", bold_font_path=""):
        self.regular = find_font(font_path)
        self.bold = find_font(bold_font_path or font_path, True)
        self.serif = str(Path(__file__).parent / "fonts/NanumMyeongjo-Regular.ttf")
        self.serif_bold = str(Path(__file__).parent / "fonts/NanumMyeongjo-ExtraBold.ttf")

    @lru_cache(maxsize=100)
    def font(self, size, bold=False, serif=False, family="auto"):
        if family not in ("auto", "sans", "serif", "bold"):
            return ImageFont.truetype(str(FONT_ROOT / FONT_BY_ID[family].path), size)
        path = (self.serif_bold if bold else self.serif) if serif else (self.bold if bold else self.regular)
        return ImageFont.truetype(path, size)

    def wrap(self, text, font, width, balanced=False):
        return wrap_text(text, font, width, balanced=balanced)

    @lru_cache(maxsize=20)
    def font_preview(self, family):
        canvas = Image.new("RGB", (800, 130), "#F6F4ED")
        self.text(ImageDraw.Draw(canvas), "우리 가게의 특별한 순간", (24, 27, 752, 88), "#243B30",
                  48, 24, family == "bold", serif=family == "serif", family=family)
        output = BytesIO(); canvas.save(output, "PNG")
        return output.getvalue()

    def text(self, draw, text, box, color, size=48, minimum=20, bold=False, align="left", *, serif=False, balanced=False, leading=1.35, family="auto"):
        if not text:
            return 0
        x, y, w, h = box
        for value in range(size, minimum - 1, -1):
            font = self.font(value, bold, serif, family)
            lines = self.wrap(text, font, w, balanced)
            step = round(value * leading)
            if len(lines) * step <= h and all(font.getlength(line) <= w for line in lines):
                for index, line in enumerate(lines):
                    offset = (w - font.getlength(line)) / 2 if align == "center" else 0
                    draw.text((x+offset, y+index*step), line, font=font, fill=color, anchor="lt")
                return len(lines) * step
        raise LayoutError("문구가 포스터 영역에 맞지 않습니다. 글귀를 조금 줄여 주세요.")

    def decorative_type(self, canvas, value, box, color, family, *, outline=False):
        x,y,w,h=map(int,box)
        mask=Image.new("L",(w,h))
        self.text(ImageDraw.Draw(mask),value,(4,4,w-8,h-8),255,240,30,True,"center",
                  family=family,balanced=True,leading=1.02)
        if outline:
            mask=ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(7)),mask)
        canvas.paste(color,(x,y,x+w,y+h),mask)

    def photo(self, canvas, data, box, panel, ink, product, fit="contain", radius=10, angle=0, rotate_tall=False):
        x, y, w, h = map(int, box)
        if min(w, h) < 100:
            raise LayoutError("상품 사진을 배치할 공간이 부족합니다. 글귀를 줄여 주세요.")
        frame = Image.new("RGB", (w, h), panel)
        if data:
            source = Image.open(BytesIO(data)).convert("RGBA")
            if has_transparency(source):
                bounds = source.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox()
                if bounds:
                    source = source.crop(bounds)
                if angle and (rotate_tall or source.width/source.height > 1.3):
                    source = source.rotate(angle, Image.Resampling.BICUBIC, expand=True)
                # Transparent assets float on the layout, with breathing room for a soft shadow.
                scale = min(w*.88/source.width, h*.86/source.height)
                source = source.resize((max(1, round(source.width*scale)), max(1, round(source.height*scale))), Image.Resampling.LANCZOS)
                layer = Image.new("RGBA", (w, h))
                px, py = (w-source.width)//2, (h-source.height)//2
                shadow_mask = Image.new("L", (w, h))
                shadow_mask.paste(source.getchannel("A"), (px+7, py+17))
                shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(18)).point(lambda v: round(v*.23))
                shadow = Image.new("RGBA", (w, h), (24, 20, 17, 0))
                shadow.putalpha(shadow_mask)
                layer.alpha_composite(shadow)
                layer.alpha_composite(source, (px, py))
                canvas.paste(layer, (x, y), layer)
                return
            if fit == "cover":
                source = ImageOps.fit(source, (w, h), method=Image.Resampling.LANCZOS)
                frame.paste(source, (0, 0), source)
            else:
                source = ImageOps.contain(source, (int(w*.94), int(h*.94)), Image.Resampling.LANCZOS)
                frame.paste(source, ((w-source.width)//2, (h-source.height)//2), source)
        else:
            d = ImageDraw.Draw(frame)
            d.ellipse((w*.12, h*.10, w*.88, h*.9), outline=ink, width=2)
            self.text(d, product, (w*.2, h*.25, w*.6, h*.5), ink, 50, 18, True, "center")
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, w-1, h-1), radius=radius, fill=255)
        canvas.paste(frame, (x, y), mask)

    def price(self, draw, brief, box, fg, accent, muted, *, large=False):
        if brief.price is None:
            return
        x, y, w, h = box
        sale_height = h-47 if brief.original_price is not None else h
        self.text(draw, f"{brief.price:,}원", (x, y, w, sale_height), accent,
                  128 if large else 66, 28, True, leading=1.15)
        if brief.original_price is not None:
            label = f"{brief.original_price:,}원"
            font = self.font(24)
            length = font.getlength(label)
            py = y+sale_height+4
            draw.text((x, py), label, font=font, fill=muted, anchor="lt")
            draw.line((x, py+13, x+length, py+13), fill=muted, width=1)
            discount = (brief.original_price-brief.price)*100//brief.original_price
            self.text(draw, f"{discount}% 할인", (x+length+24, py-2, w-length-24, 38),
                      fg, 25, 18, True)

    def headline_family(self, design):
        if design.typography != "auto":
            return design.typography
        return default_font(design.template_id)

    def render(self, brief, design, product_image=None, logo=None):
        try:
            return self._render_design(brief, design, product_image, logo)
        except LayoutError:
            # Dense factual inputs need more room, not truncated product details.
            return self._render_compact(brief, design, product_image, logo)

    def _render_design(self, brief, design, product_image=None, logo=None):
        bg, fg, accent, panel, muted = PALETTES[design.palette]
        canvas = Image.new("RGB", (WIDTH, HEIGHT), bg)
        d = ImageDraw.Draw(canvas)
        variant = design.template_id
        if variant == "spotlight":
            glow = Image.new("L", (108, 135))
            glow.putdata([round(90*math.exp(-((x-54)**2/1050+(y-57)**2/1300))) for y in range(135) for x in range(108)])
            glow = glow.resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)
            canvas.paste(Image.new("RGB", (WIDTH, HEIGHT), panel), (0, 0), glow)
        offer = variant in ("bold_sale", "split")
        bottom_price = brief.price is not None and not offer
        detail = "\n".join(x for x in [brief.product, brief.message] if x)
        detail_font = 25
        detail_height = len(self.wrap(detail, self.font(detail_font), 936))*34
        price_height = (176 if design.emphasis == "price" else 128) if bottom_price else 0
        period_height = 40 if brief.period else 0
        facts_y = 1222-detail_height-period_height-price_height-16
        top_end = facts_y-30
        if top_end < 790:
            detail_font = 21
            detail_height = len(self.wrap(detail, self.font(detail_font), 936))*29
            facts_y = 1222-detail_height-period_height-price_height-16
            top_end = facts_y-30
        if top_end < 600:
            raise LayoutError("문구와 상품 사진을 함께 배치하기 어렵습니다. 메시지를 줄여 주세요.")
        required = {"split": 860 if brief.price is not None else 680,
                    "soft_frame": 705, "market": 720, "type_stack": 730,
                    "spotlight": 710, "magazine": 760, "diagonal": 740,
                    "floating_type": 730, "big_focus": 730}.get(variant, 600)
        if top_end < required:
            raise LayoutError("이 구도에 문구가 모두 들어가지 않습니다. 메시지를 줄여 주세요.")
        family = self.headline_family(design)
        serif, bold = family == "serif", family == "bold"

        def title(box, size=86, align="left"):
            return self.text(d, design.headline, box, fg, size, 30, bold, align,
                             serif=serif, balanced=True, leading=1.18, family=family)

        def subtitle(box, align="left"):
            return self.text(d, design.subhead, box, muted, 29, 19, False, align)

        def photo(box, radius=12):
            self.photo(canvas, product_image, box, panel, fg, brief.product, design.image_fit, radius)

        # Store identity and logo occupy a fixed safe area in every composition.
        self.text(d, brief.store.name, (72, 56, 750, 52), fg, 27, 18, True)
        if logo:
            im = Image.open(BytesIO(logo)).convert("RGBA")
            im.thumbnail((150, 65), Image.Resampling.LANCZOS)
            canvas.paste(im, (1008-im.width, 50), im)
        else:
            d.ellipse((981, 65, 1005, 89), fill=accent)

        if variant == "editorial":
            d.line((72, 122, 1008, 122), fill=muted, width=1)
            used = title((72, 164, 936, 210), 92)
            sub_y = 164+used+24
            sub_used = subtitle((76, sub_y, 850, 82))
            photo_y = sub_y+sub_used+24
            photo((65, photo_y, 950, top_end-photo_y))
        elif variant == "bold_sale":
            d.line((72, 125, 1008, 125), fill=fg, width=3)
            title((72, 155, 936, 168), 88)
            subtitle((76, 335, 890, 78))
            if brief.price is not None:
                self.price(d, brief, (72, 430, 936, 200), fg, accent, muted, large=True)
                photo_top = 630
            else:
                photo_top = 430
            # A broad colour field grounds the product; no sticker obscures it.
            d.rounded_rectangle((72, photo_top, 1008, top_end), radius=38, fill=panel)
            photo((78, photo_top+4, 924, top_end-photo_top-8), 32)
        elif variant == "split":
            d.rounded_rectangle((542, 146, 1008, top_end), radius=220, fill=panel)
            title((72, 174, 430, 280), 79)
            subtitle((76, 477, 414, 155))
            if brief.price is not None:
                self.price(d, brief, (72, max(650, top_end-220), 430, 180),
                           fg, accent, muted, large=True)
            photo((548, 160, 454, top_end-175), 180)
        elif variant == "soft_frame":
            photo_end = max(420, top_end-285)
            d.rounded_rectangle((192, 142, 888, photo_end), radius=230, fill=panel)
            photo((170, 145, 740, photo_end-145), 180)
            title((72, photo_end+33, 936, 155), 77, "center")
            subtitle((130, photo_end+202, 820, 78), "center")
        elif variant == "night":
            d.rectangle((29, 29, 1050, 1320), outline=accent, width=1)
            d.line((72, 125, 1008, 125), fill=accent, width=1)
            used = title((72, 181, 560, 300), 86)
            subtitle((76, 510, 340, min(180, top_end-510)))
            # Deliberately lower/right composition leaves a quiet column of copy.
            photo_y = max(367, 181+used+24)
            photo((434, photo_y, 574, top_end-photo_y))
            d.line((76, top_end-16, 274, top_end-16), fill=accent, width=3)
        elif variant == "type_stack":
            title((72, 140, 936, 95), 58, "center")
            band_height = min(190, (top_end-250)//3)
            for row in range(3):
                y = 250 + row*band_height
                self.text(d, brief.product, (40, y, 1000, band_height), fg, 185, 30, True, "center",
                          family=family, balanced=True, leading=1.05)
            photo((72, 246, 936, top_end-246))
            # Product details below remain fully readable; background words are decorative.
        elif variant == "floating_type":
            title((72, 140, 936, 110), 58, "center")
            band_h=(top_end-270)//2
            for y in (270,270+band_h):
                self.decorative_type(canvas,brief.product,(28,y,1024,band_h),fg,family)
            self.photo(canvas,product_image,(140,240,800,top_end-240),panel,fg,
                       brief.product,design.image_fit,12,angle=-13,rotate_tall=True)
            # A thin outline echoes the lower word without an opaque block over the product.
            if product_image and has_transparency(Image.open(BytesIO(product_image)).convert("RGBA")):
                self.decorative_type(canvas,brief.product,(28,270+band_h,1024,band_h),fg,family,outline=True)
        elif variant == "big_focus":
            title((72,140,936,105),64)
            self.decorative_type(canvas,brief.product,(30,255,1020,min(340,top_end-390)),fg,family)
            # The source stays whole by default; a deliberate cover edit can crop opaque photos.
            photo((155,370,870,top_end-370),8)
            d.line((72,top_end-28,132,top_end-28),fill=fg,width=3)
        elif variant == "magazine":
            d.line((72, 126, 1008, 126), fill=muted, width=1)
            title((72, 165, 402, 300), 100)
            subtitle((76, 500, 330, min(180, top_end-520)))
            photo((494, 215, 514, top_end-215))
            d.line((76, top_end-20, 235, top_end-20), fill=accent, width=2)
        elif variant == "spotlight":
            title((72, 150, 936, 150), 100, "center")
            subtitle((140, 306, 800, 80), "center")
            d.ellipse((180, top_end-78, 900, top_end-6), fill=panel)
            d.line((180, top_end-40, 900, top_end-40), fill=muted, width=1)
            photo((130, 378, 820, top_end-378))
        elif variant == "diagonal":
            d.rectangle((565, 136, 1008, top_end), fill=panel)
            title((72, 161, 936, 156), 94)
            subtitle((76, 333, 800, 78))
            label = Image.new("RGBA", (max(100, round(top_end-430)), 145))
            self.text(ImageDraw.Draw(label), brief.product, (0, 0, label.width, 140), fg, 105, 24,
                      True, family="dohyeon", leading=1.1)
            label = label.transpose(Image.Transpose.ROTATE_90)
            canvas.paste(label, (825, 420), label)
            self.photo(canvas, product_image, (65, 422, 900, top_end-422), panel, fg,
                       brief.product, design.image_fit, 12, angle=-12)
        else:  # market: the product leads, copy follows.
            photo_end = max(440, top_end-280)
            d.ellipse((202, 147, 878, photo_end), fill=panel)
            photo((72, 140, 936, photo_end-140))
            title((72, photo_end+28, 936, 158), 82)
            subtitle((76, photo_end+202, 910, 76))

        y = facts_y
        if bottom_price:
            self.price(d, brief, (72, y, 936, price_height), fg, accent, muted,
                       large=design.emphasis == "price")
            y += price_height
        self.text(d, detail, (72, y, 936, detail_height), fg, detail_font, detail_font)
        y += detail_height+12
        if brief.period:
            self.text(d, brief.period, (72, y, 936, period_height), muted, 23, 18)
        d.line((72, 1247, 1008, 1247), fill=muted, width=1)
        self.text(d, design.cta, (72, 1270, 500, 43), fg, 25, 18, True)
        if not design.hide_footer:
            self.text(d, brief.contact or brief.store.region, (600, 1266, 408, 53), muted, 22, 16)
        output = BytesIO()
        canvas.save(output, "PNG", optimize=True)
        return output.getvalue()

    def _render_compact(self, brief, design, product_image=None, logo=None):
        bg, fg, accent, panel, muted = PALETTES[design.palette]
        canvas = Image.new("RGB", (WIDTH, HEIGHT), bg)
        d = ImageDraw.Draw(canvas)
        family = self.headline_family(design)
        group = next(t.family for t in TEMPLATES if t.id == design.template_id)
        detail = "\n".join([brief.product, " ".join(brief.message.split())]).strip()
        footer = "" if design.hide_footer else (brief.contact or brief.store.region)
        footer_h = 40+len(self.wrap(footer, self.font(18), 936))*25 if footer else 40
        footer_y = HEIGHT-42-footer_h
        period_h = len(self.wrap(brief.period, self.font(20), 936))*28 if brief.period else 0
        price_h = 126 if brief.price is not None else 0
        for body_size in (23, 21, 19, 18):
            detail_h = len(self.wrap(detail, self.font(body_size), 936))*round(body_size*1.35)
            facts_y = footer_y-24-period_h-detail_h-price_h-20
            if facts_y >= 620:
                break
        if facts_y < 620:
            raise LayoutError("입력한 문구가 너무 깁니다. 메시지를 조금 줄여 주세요.")
        self.text(d, brief.store.name, (72, 54, 750, 55), fg, 27, 18, True)
        if logo:
            im=Image.open(BytesIO(logo)).convert("RGBA");im.thumbnail((150,65))
            canvas.paste(im,(1008-im.width,50),im)
        d.line((72,126,1008,126),fill=muted,width=1)
        title_args = {"serif": family == "serif", "family": family, "balanced": True, "leading": 1.2}
        if group == "offer":
            self.text(d, design.headline, (72, 165, 420, 225), fg, 74, 24, family=="bold", **title_args)
            self.text(d, design.subhead, (76, 410, 410, facts_y-440), muted, 25, 18)
            self.photo(canvas,product_image,(540,160,468,facts_y-185),panel,fg,brief.product,design.image_fit,30)
        elif group == "brand":
            self.text(d,design.headline,(72,165,936,142),fg,70,24,family=="bold",**title_args)
            self.text(d,design.subhead,(76,345,320,facts_y-375),muted,25,18)
            self.photo(canvas,product_image,(435,325,573,facts_y-350),panel,fg,brief.product,design.image_fit,24)
        else:
            used=self.text(d,design.headline,(72,160,936,150),fg,74,24,family=="bold",**title_args)
            used_sub=self.text(d,design.subhead,(76,175+used,920,80),muted,25,18)
            photo_y=195+used+used_sub
            self.photo(canvas,product_image,(72,photo_y,936,facts_y-photo_y-24),panel,fg,brief.product,design.image_fit,24)
        y=facts_y
        if brief.price is not None:
            self.price(d,brief,(72,y,936,price_h),fg,accent,muted,large=True);y+=price_h
        self.text(d,detail,(72,y,936,detail_h),fg,body_size,body_size);y+=detail_h+12
        if brief.period:
            self.text(d,brief.period,(72,y,936,period_h),muted,20,20)
        d.line((72,footer_y,1008,footer_y),fill=muted,width=1)
        self.text(d,design.cta,(72,footer_y+12,936,36),fg,24,18,True)
        if footer:
            self.text(d,footer,(72,footer_y+49,936,footer_h-34),muted,18,18)
        output=BytesIO();canvas.save(output,"PNG",optimize=True)
        return output.getvalue()
