"""Write synthetic Korean store reviews for stage-2 training stores with an API model.

Usage:
    python -m scrable.synth_reviews --docs data/train_store_docs --out data/dpo_reviews.jsonl \
        --avoid data/store_reviews.jsonl

Each first-level folder under --docs is one store (same layout as Config.documents_dir).
The output is synthetic data: say so wherever it is used. Requests are cached, so a rerun
with the same arguments costs nothing and resumes after an interruption.
"""

import argparse
import difflib
import json
import random
from pathlib import Path

from .config import Config
from .data import load_custom_reviews, load_documents, normalize
from .llm import CachedLLM, make_backend, structured_call

# Deliberately more complaints than real review sites: the small model is weakest on them,
# so they carry most of the preference signal.
RATING_COUNTS = {5: 24, 4: 14, 3: 12, 2: 14, 1: 16}
BATCH = 8

SITUATIONS = {
    5: [
        "특정 메뉴의 맛을 구체적으로 칭찬",
        "단골이라 자주 시킨다는 내용",
        "포장이 꼼꼼했다는 칭찬",
        "생각보다 배달이 빨랐다는 칭찬",
        "리뷰 이벤트에 참여해 서비스를 받았다는 내용",
        "가격 대비 양이 많다는 칭찬",
        "야식·아이 간식·모임처럼 먹은 상황을 곁들인 칭찬",
        "아주 짧은 한 줄 칭찬",
    ],
    4: [
        "칭찬하면서 '그래도' 또는 '다만'으로 작은 아쉬움을 덧붙임",
        "맛은 좋은데 가격이 조금 아쉬움",
        "맛은 좋은데 양이 조금 아쉬움",
        "간이나 맵기에 대한 개인 취향 의견",
        "배달이 조금 늦었지만 맛은 괜찮았음",
    ],
    3: [
        "무난하지만 특별하지 않다는 평",
        "기대보다 별로였다는 평",
        "한 메뉴는 좋고 다른 메뉴는 아쉬움",
        "가게 정보를 보면 답할 수 있는 질문이 섞인 리뷰(기본 제공 여부, 휴무일, 옵션 등)",
        "음식이 식거나 눅눅하게 도착",
    ],
    2: [
        "배달이 많이 늦었다는 불만(빈정대는 '^^' 반어 표현 사용)",
        "요청사항이 반영되지 않음",
        "포장이 새거나 흐트러져서 도착",
        "옵션이 잘못 들어감",
        "품절이나 재료 소진으로 주문이 취소됨",
        "전화 연결이 안 돼서 답답했음",
    ],
    1: [
        "주문한 메뉴가 빠져서 옴",
        "음식에서 이물질(머리카락, 비닐 등)이 나옴",
        "덜 익었거나 상한 것 같다는 불만",
        "휴무일이나 브레이크타임을 몰라 헛걸음하거나 주문 실패",
        "건강·알레르기 관련 요청이 무시됨",
        "다시는 안 시킨다는 강한 불만",
    ],
}

STYLES = [
    "짧고 담백하게",
    "ㅎㅎ, ㅠㅠ, !! 같은 표현을 자연스럽게 섞어서",
    "주문한 배경(언제, 누구와, 왜)으로 시작해서",
    "구어체 반말 섞인 말투로",
    "두세 문장으로 조금 길게",
]

SYSTEM = """당신은 한국 배달앱 리뷰 연구용 합성 데이터를 작성합니다.
주어진 가게 정보와 별점, 상황에 맞는 고객 리뷰를 실제 고객이 쓴 것처럼 작성하세요.
- 리뷰마다 지정된 상황과 말투를 따르고, 서로 다른 메뉴와 표현을 쓰세요.
- 가게 정보에 있는 메뉴·정책과 모순되지 않게 쓰세요. 고객은 가게 정책을 잘 모를 수 있습니다.
- 실제 사람 이름, 전화번호, 주소, 주문번호는 넣지 마세요.
- 리뷰 문장만 쓰고 별점이나 설명은 붙이지 마세요.
출력은 {"reviews": ["리뷰1", "리뷰2", ...]} 형식의 JSON 객체 하나뿐입니다."""


def validator(count):
    def check(value):
        items = value.get("reviews")
        if set(value) != {"reviews"} or not isinstance(items, list) or len(items) != count:
            raise ValueError(f'Expected {{"reviews": [...]}} with exactly {count} strings')
        if not all(isinstance(x, str) and 3 <= len(x.strip()) <= 300 for x in items):
            raise ValueError("Each review must be a string of 3 to 300 characters")
        return [x.strip() for x in items]

    return check


def plan(stores, per_store_counts=RATING_COUNTS, seed=42):
    """Deterministic (store, rating, [(situation, style), ...]) request batches."""
    rng = random.Random(seed)
    batches = []
    for store in stores:
        for rating, count in per_store_counts.items():
            specs = [(rng.choice(SITUATIONS[rating]), rng.choice(STYLES)) for _ in range(count)]
            for start in range(0, count, BATCH):
                batches.append((store, rating, specs[start : start + BATCH]))
    return batches


def too_similar(text, seen, threshold=0.85):
    key = normalize(text)
    return any(
        key == other or difflib.SequenceMatcher(None, key, other).ratio() >= threshold for other in seen
    )


def generate(llm, documents, batches, avoid=(), max_tokens=2048, attempts=3):
    info = {}
    for doc in documents:
        info.setdefault(doc.app_id, []).append(doc.text)
    seen = [normalize(t) for t in avoid]
    rows, dropped = [], 0
    counters = {}
    for store, rating, specs in batches:
        if store not in info:
            raise ValueError(f"No store documents for {store}")
        payload = {
            "store_info": "\n\n".join(info[store]),
            "rating": rating,
            "reviews_to_write": [{"situation": s, "style": t} for s, t in specs],
        }
        texts = structured_call(
            llm,
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            validator(len(specs)),
            max_tokens,
            attempts,
        )
        for text in texts:
            if too_similar(text, seen):
                dropped += 1
                continue
            seen.append(normalize(text))
            counters[store] = counters.get(store, 0) + 1
            rows.append(
                {
                    "id": f"dpo-{store}-{counters[store]:03d}",
                    "app_id": store,
                    "rating": rating,
                    "review": text,
                    "expert_response": None,
                }
            )
    return rows, dropped


def main():
    parser = argparse.ArgumentParser(description="Write synthetic store reviews for stage-2 training stores")
    parser.add_argument("--docs", required=True, help="Folder with one subfolder of documents per store")
    parser.add_argument("--out", required=True, help="Output JSONL")
    parser.add_argument("--avoid", nargs="*", default=[], help="JSONL files whose reviews must not be copied")
    parser.add_argument("--backend", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--cache", default="outputs/synth_reviews_cache")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    documents = load_documents(args.docs, "default")
    stores = sorted({d.app_id for d in documents} - {"default"})
    if not stores:
        raise ValueError(f"Put each store's documents in its own subfolder of {args.docs}")
    avoid = [r.review for path in args.avoid for r in load_custom_reviews(path)]
    llm = CachedLLM(make_backend(args.backend, args.model, Config()), args.cache)
    batches = plan(stores, seed=args.seed)
    print(f"{len(stores)} stores, {len(batches)} requests to {args.backend}:{args.model}", flush=True)
    rows, dropped = generate(llm, documents, batches, avoid)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} reviews to {out} ({dropped} near-duplicates dropped; API calls {llm.calls})")


if __name__ == "__main__":
    main()
