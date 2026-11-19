"""Geometry hints, never unverified product facts or image-derived advertising claims."""
from io import BytesIO
import re
from PIL import Image
from .product_image import has_transparency


def is_product_set(product):
    return bool(re.search(r"세트|묶음|모둠|모듬|패키지|\bset\b|\bbundle\b", product, re.I))


def visual_profile(data, brief):
    profile = {"has_image": bool(data), "has_price": brief.price is not None,
               "multiple_items": is_product_set(brief.product), "shape": "none", "background": "none"}
    if data:
        image = Image.open(BytesIO(data)).convert("RGBA")
        cutout = has_transparency(image)
        bounds = image.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox() if cutout else None
        if bounds:
            image = image.crop(bounds)
        ratio = image.width/image.height
        profile.update(shape="tall" if ratio < .65 else "wide" if ratio > 1.8 else "balanced",
                       aspect_ratio=round(ratio, 2), background="cutout" if cutout else "photo")
    return profile
