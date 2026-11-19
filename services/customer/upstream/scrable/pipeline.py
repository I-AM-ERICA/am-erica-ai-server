import json
import math
import random
from concurrent.futures import ThreadPoolExecutor

from .llm import structured_call
from .prompts import GROUNDING_RULES, PHOTO_RUBRIC_NOTES, PHOTO_RULES, PROMPT_GEN_SYSTEM, judge_system, rubric
from .rag import context_payload
from .types import CRITERIA, Feedback

DOCUMENT_CRITERIA = ("specificity", "accuracy")


def generation_messages(prompt, review, chunks, channel="review", photo_description=None):
    """Exact chat input of the response generator; stage-2 training reuses it unchanged.

    channel="email" puts the text under "email" instead of "review"; nothing else changes.
    photo_description (photo reviews) adds that field and PHOTO_RULES; without it the input is
    exactly the training input.
    """
    # Expert reference is deliberately absent from response-generation inputs.
    payload = {
        "app_id": review.app_id,
        channel: review.review,
        "retrieved_documents": context_payload(chunks),
    }
    if review.rating is not None:
        payload["rating"] = review.rating
    system = prompt + "\n\n" + GROUNDING_RULES
    if photo_description:
        payload["photo_description"] = photo_description
        system += "\n\n" + PHOTO_RULES
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


class ScoredResponseGenerator:
    def __init__(self, response_llm, judge_llm, retriever, config, describer=None):
        self.response_llm = response_llm
        self.judge_llm = judge_llm
        self.retriever = retriever
        self.config = config
        self.describer = describer  # PhotoDescriber (scrable/photos.py) or None
        self.photo_descriptions = {}

    def generate(self, prompt, review):
        chunks = self.retriever.retrieve(review.review, review.app_id, self.config.top_k)
        description = None
        if review.image and self.describer is not None:
            description = self.describer.describe(review.image)
            self.photo_descriptions[review.id] = description
        prediction = self.response_llm.complete(
            generation_messages(prompt, review, chunks, self.config.channel, description),
            max_tokens=self.config.response_tokens,
        )
        if not prediction.strip():
            raise ValueError("Empty generated response")
        return prediction, chunks

    def judge_criterion(self, review, prediction, chunks, criterion):
        channel = self.config.channel
        note = PHOTO_RUBRIC_NOTES.get(criterion, "") if review.image else ""
        payload = {
            "criterion": criterion,
            "rubric": rubric(criterion, channel) + note,
            "app_id": review.app_id,
            channel: review.review,
            "prediction": prediction,
        }
        if review.rating is not None:
            payload["rating"] = review.rating
        # Specificity and accuracy see the retrieved store documents, so facts taken from them are
        # not penalized as invented. Only accuracy sees the expert answer (paper's setup).
        if criterion in DOCUMENT_CRITERIA:
            payload["retrieved_documents"] = context_payload(chunks)
        if criterion == "accuracy":
            payload["expert_response"] = review.expert_response
        content = json.dumps(payload, ensure_ascii=False)
        if review.image:
            # The judge sees the customer's photo itself (Claude image input), not the description.
            if self.config.role("judge")[0] != "anthropic":
                raise ValueError("Photo reviews need judge_backend=anthropic (the judge must see the photo)")
            from .photos import image_block

            content = [
                image_block(review.image),
                {"type": "text", "text": "첨부된 사진은 손님이 리뷰에 올린 사진입니다.\n" + content},
            ]
        return structured_call(
            self.judge_llm,
            [
                {"role": "system", "content": judge_system(channel)},
                {"role": "user", "content": content},
            ],
            validate_judgment,
            self.config.judge_tokens,
            self.config.json_attempts,
        )

    def _map(self, function, items):
        workers = min(self.config.judge_workers, len(items))
        if workers <= 1:
            return [function(item) for item in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(function, items))

    @staticmethod
    def feedback(review, prediction, chunks, results, photo_description=None):
        scores, rationales, suggestions = {}, {}, []
        for criterion in CRITERIA:  # Fixed order keeps results reproducible under parallel judging.
            result = results[criterion]
            scores[criterion] = result["score"]
            rationales[criterion] = result["reason"]
            suggestions.extend(f"[{criterion}] {item}" for item in result["suggestions"])
        return Feedback(
            review.id,
            review.review,
            review.app_id,
            prediction,
            scores,
            suggestions,
            rationales,
            [c.id for c in chunks],
            photo_description,
        )

    def judge(self, review, prediction, chunks):
        results = self._map(lambda c: self.judge_criterion(review, prediction, chunks, c), list(CRITERIA))
        return self.feedback(
            review, prediction, chunks, dict(zip(CRITERIA, results)), self.photo_descriptions.get(review.id)
        )

    def __call__(self, prompt, reviews, description="ScoredResponseGen"):
        from tqdm.auto import tqdm

        if not reviews:
            raise ValueError("ScoredResponseGen requires nonempty reviews")
        # Generate sequentially (one local GPU model), then judge every review x criterion.
        # Seconds per reply are kept for the speed report (a cached reply takes ~0 s).
        import time

        generated, self.generation_seconds = [], []
        for review in tqdm(reviews, desc=f"{description}: generate"):
            start = time.perf_counter()
            generated.append(self.generate(prompt, review))
            self.generation_seconds.append(time.perf_counter() - start)
        tasks = [(i, c) for i in range(len(reviews)) for c in CRITERIA]
        progress = tqdm(total=len(tasks), desc=f"{description}: judge")

        def run(task):
            i, criterion = task
            result = self.judge_criterion(reviews[i], *generated[i], criterion)
            progress.update(1)
            return result

        try:
            results = self._map(run, tasks)
        finally:
            progress.close()
        by_review = [{} for _ in reviews]
        for (i, criterion), result in zip(tasks, results):
            by_review[i][criterion] = result
        return [
            self.feedback(review, *generated[i], by_review[i], self.photo_descriptions.get(review.id))
            for i, review in enumerate(reviews)
        ]


def validate_judgment(value):
    if set(value) != {"score", "reason", "suggestions"}:
        raise ValueError("Required keys: score, reason, suggestions")
    if type(value["score"]) is not int or not 1 <= value["score"] <= 5:
        raise ValueError("score must be an integer in [1, 5]")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("reason must be a nonempty string")
    if not isinstance(value["suggestions"], list) or not all(
        isinstance(x, str) and x.strip() for x in value["suggestions"]
    ):
        raise ValueError("suggestions must be an array of nonempty strings")
    return value


def average_score(feedback):
    if not feedback:
        raise ValueError("Cannot average empty feedback")
    return sum(f.score for f in feedback) / len(feedback)


def identify_ir(feedback, lowest_fraction=0.30, random_fraction=0.10, seed=42):
    """Ceil fractions for small datasets; random cases exclude the lowest set."""
    if not feedback:
        raise ValueError("IdentifyIR requires nonempty feedback")
    if not 0 < lowest_fraction <= 1 or not 0 <= random_fraction <= 1:
        raise ValueError("Invalid selection fractions")
    if lowest_fraction + random_fraction > 1:
        raise ValueError("Selection fractions exceed 1")
    if len({f.review_id for f in feedback}) != len(feedback):
        raise ValueError("Duplicate review IDs in feedback")
    ordered = sorted(feedback, key=lambda f: (f.score, f.review_id))
    count = math.ceil(len(ordered) * lowest_fraction)
    lowest, remaining = ordered[:count], ordered[count:]
    random_count = min(len(remaining), math.ceil(len(ordered) * random_fraction))
    extra = random.Random(seed).sample(remaining, random_count)
    return lowest + extra


class PromptGenerator:
    def __init__(self, llm, config):
        self.llm, self.config = llm, config

    def __call__(self, current_prompt, selected):
        # Retain every selected case while bounding each example's length.
        payload = {
            "current_prompt": current_prompt,
            "feedback": [
                {
                    "review": f.review[:600],
                    "prediction": f.prediction[:800],
                    "score": f.score,
                    "suggestions": [s[:300] for s in f.suggestions],
                    "scores": f.scores,
                }
                for f in selected
            ],
        }
        result = structured_call(
            self.llm,
            [
                {"role": "system", "content": PROMPT_GEN_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            validate_prompt,
            self.config.prompt_tokens,
            self.config.json_attempts,
        )
        return result["prompt"].strip()


def validate_prompt(value):
    if set(value) != {"prompt"} or not isinstance(value["prompt"], str):
        raise ValueError("Expected exactly one string field: prompt")
    if not 20 <= len(value["prompt"].strip()) <= 6000:
        raise ValueError("Prompt must contain 20 to 6000 characters")
    return value
