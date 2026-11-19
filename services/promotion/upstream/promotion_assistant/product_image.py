"""Local pretrained segmentation, immutable RGB, and reusable alpha cutouts."""
from dataclasses import dataclass
from io import BytesIO
import threading
from PIL import Image, ImageChops, ImageFilter


@dataclass
class ProductImage:
    data: bytes | None
    status: str
    warning: str = ""


def png(image):
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def has_transparency(image):
    return image.getchannel("A").getextrema()[0] < 250


class ProductProcessor:
    def __init__(self, session=None):
        self.session = session
        self.lock = threading.Lock()
        self.unavailable = False

    def warmup(self):
        with self.lock:
            if self.session is None:
                # Pin the model: rembg's default is a different, separately licensed model.
                from rembg import new_session
                self.session = new_session("u2net", providers=["CPUExecutionProvider"])

    def process(self, data, mode="auto"):
        if data is None:
            return ProductImage(None, "none")
        source = Image.open(BytesIO(data)).convert("RGBA")
        if mode == "keep":
            return ProductImage(data, "kept")
        if has_transparency(source):
            return ProductImage(data, "transparent")
        try:
            if self.unavailable:
                raise RuntimeError("segmentation unavailable")
            self.warmup()
            # Use only the predicted alpha: do not redraw or recolour the product.
            with self.lock:
                mask = self.session.predict(source.convert("RGB"))[0].convert("L")
            mask = mask.resize(source.size, Image.Resampling.LANCZOS)
            histogram = mask.histogram()
            coverage = sum(histogram[32:])/(source.width*source.height)
            if not .005 < coverage < .995:
                raise ValueError("degenerate mask")
            mask = mask.filter(ImageFilter.GaussianBlur(.45))
            result = source.copy()
            result.putalpha(ImageChops.multiply(source.getchannel("A"), mask))
            return ProductImage(png(result), "removed")
        except Exception:
            if mode == "remove":
                raise ValueError("상품 배경을 분리하지 못했습니다. '원본 배경 유지'로 다시 시도하거나 배경이 단순한 사진을 올려 주세요.") from None
            return ProductImage(data, "kept", "배경을 안정적으로 분리하지 못해 원본 사진을 사용했습니다. 필요하면 '원본 배경 유지'를 선택해 주세요.")
