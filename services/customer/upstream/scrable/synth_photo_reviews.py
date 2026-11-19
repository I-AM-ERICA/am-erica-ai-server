"""Write synthetic customer reviews for photos, for the photo-review test (plan A).

    python -m scrable.synth_photo_reviews --photos data/review_photos --docs data/store_docs \
        --out data/photo_reviews.jsonl

Put the photos in one subfolder per store, named like the store-document folders:
data/review_photos/bunsik-01/tteokbokki.jpg, data/review_photos/chicken-01/box.png, ...
Claude sees each photo and the store information, and writes the review text and star rating a
customer who took that photo might post. Use only photos you may use (your own, or with
permission): never real customers' photos. The reviews are synthetic: say so wherever used.
Requests are cached, so a rerun costs nothing.

--style picks what kind of review to write for each photo (several styles = one row per style):
  photo      the review describes what the photo shows (first test; the text repeats the photo)
  short      one short line that barely mentions the food, like many real photo reviews, so the
             photo carries information the text does not
  complaint  a complaint the photo cannot show (cold, late, too salty, request ignored), 1-3 stars:
             a good reply must not claim the photo shows the problem

    python -m scrable.synth_photo_reviews --photos data/review_photos --docs data/store_docs \
        --out data/photo_reviews_v2.jsonl --style short complaint
"""

import argparse
import json
from pathlib import Path

from .config import Config
from .data import load_documents
from .llm import CachedLLM, make_backend, structured_call
from .photos import MEDIA_TYPES, image_block

SYSTEM = """당신은 한국 배달앱 사진 리뷰 연구용 합성 데이터를 작성합니다.
첨부된 사진은 손님이 리뷰에 올린 사진이라고 가정합니다. 가게 정보와 사진을 보고, 이 사진을 찍은 손님이 쓸 법한 리뷰를 작성하세요.
- 사진에 문제(양 부족, 포장 흐트러짐, 이물질, 눅눅함 등)가 보이면 그에 맞게 낮은 별점과 불만을, 문제가 없어 보이면 높은 별점과 칭찬을 쓰세요.
- 리뷰는 사진 속 내용을 한 가지 이상 언급하고, 1~3문장으로 실제 손님처럼 자연스럽게 쓰세요.
- 가게 정보와 모순되는 메뉴나 정책을 지어내지 마세요. 사람 이름, 전화번호, 주소, 주문번호는 넣지 마세요.
출력은 {"rating": 1~5 정수, "review": "리뷰 문장"} 형식의 JSON 객체 하나뿐입니다."""

# Extra instruction per style, appended to SYSTEM, and the ratings each style must have. "photo"
# adds nothing, so its requests (and cache) stay exactly those of the first test.
STYLES = {
    "photo": ("", range(1, 6)),
    "short": (
        "\n\n[이번 리뷰의 형식] 위 규칙과 달리 사진 속 음식 이름이나 모양은 말하지 말고, "
        "'잘 먹었어요', '맛있어요 또 시킬게요'처럼 아주 짧은 한 문장(20자 안팎)으로만 쓰세요. 별점은 4나 5입니다.",
        range(4, 6),
    ),
    "complaint": (
        "\n\n[이번 리뷰의 형식] 위 규칙과 달리 사진으로는 알 수 없는 불만 하나(음식이 식어서 옴, 배달이 늦음, "
        "너무 짬, 요청사항 미반영 중 하나)를 쓰세요. 사진 속 음식 이름은 한 번 말해도 되지만, "
        "사진에 그 문제가 보인다고 쓰지는 마세요. 별점은 1~3입니다.",
        range(1, 4),
    ),
}


def validate(value):
    if set(value) != {"rating", "review"}:
        raise ValueError('Expected {"rating": int, "review": str}')
    if type(value["rating"]) is not int or not 1 <= value["rating"] <= 5:
        raise ValueError("rating must be an integer from 1 to 5")
    if not isinstance(value["review"], str) or not 3 <= len(value["review"].strip()) <= 300:
        raise ValueError("review must be a string of 3 to 300 characters")
    return {"rating": value["rating"], "review": value["review"].strip()}


def validator(style):
    allowed = STYLES[style][1]

    def check(value):
        result = validate(value)
        if result["rating"] not in allowed:
            raise ValueError(f"{style} reviews need a rating in {list(allowed)}")
        return result

    return check


def style_of(review_id):
    """Style encoded in a photo-review id: photo-<store>-<name>[-short|-complaint]."""
    last = review_id.rsplit("-", 1)[-1]
    return last if last in STYLES and last != "photo" else "photo"


def find_photos(root):
    """[(store, path)] for every image in a store subfolder, in a stable order."""
    root = Path(root)
    photos = [
        (path.parent.name, path)
        for path in sorted(root.glob("*/*"))
        if path.is_file() and path.suffix.lower() in MEDIA_TYPES
    ]
    if not photos:
        raise ValueError(f"No photos in {root}/<store>/ ({', '.join(sorted(MEDIA_TYPES))})")
    return photos


def generate(llm, documents, photos, max_tokens=512, attempts=3, styles=("photo",)):
    unknown = [s for s in styles if s not in STYLES]
    if unknown:
        raise ValueError(f"Unknown style(s) {unknown}: use {sorted(STYLES)}")
    info = {}
    for doc in documents:
        info.setdefault(doc.app_id, []).append(doc.text)
    rows = []
    for style in styles:
        for store, path in photos:
            if store not in info:
                raise ValueError(f"No store documents for photo folder {store}")
            text = json.dumps({"store_info": "\n\n".join(info[store])}, ensure_ascii=False)
            result = structured_call(
                llm,
                [
                    {"role": "system", "content": SYSTEM + STYLES[style][0]},
                    {"role": "user", "content": [image_block(path), {"type": "text", "text": text}]},
                ],
                validator(style),
                max_tokens,
                attempts,
            )
            suffix = "" if style == "photo" else f"-{style}"
            rows.append(
                {
                    "id": f"photo-{store}-{path.stem}{suffix}",
                    "app_id": store,
                    "rating": result["rating"],
                    "review": result["review"],
                    "image": path.as_posix(),
                    "expert_response": None,
                }
            )
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Two photos in one store folder share a file name (ignoring the extension)")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Write synthetic reviews for photos for the photo-review test"
    )
    parser.add_argument("--photos", required=True, help="Folder with one subfolder of photos per store")
    parser.add_argument("--docs", required=True, help="Store documents (same store folder names)")
    parser.add_argument("--out", required=True, help="Output JSONL (store_reviews.jsonl format + image)")
    parser.add_argument(
        "--style", nargs="+", default=["photo"], choices=sorted(STYLES), help="Review style(s)"
    )
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--cache", default="outputs/synth_photo_reviews_cache")
    args = parser.parse_args()

    photos = find_photos(args.photos)
    llm = CachedLLM(make_backend("anthropic", args.model, Config()), args.cache)
    print(
        f"{len(photos)} photos x {len(args.style)} style(s), requests to anthropic:{args.model}", flush=True
    )
    rows = generate(llm, load_documents(args.docs, "default"), photos, styles=args.style)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    ratings = {r: sum(row["rating"] == r for row in rows) for r in range(1, 6)}
    print(f"Wrote {len(rows)} photo reviews to {out} (ratings {ratings}; API calls {llm.calls})")


if __name__ == "__main__":
    main()
