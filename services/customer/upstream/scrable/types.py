import math
from dataclasses import asdict, dataclass, field

CRITERIA = ("relevancy", "specificity", "accuracy", "grammar")
WEIGHTS = {"relevancy": 1, "specificity": 1, "accuracy": 2, "grammar": 1}


@dataclass(frozen=True)
class Review:
    id: str
    review: str
    app_id: str
    expert_response: str | None = None
    rating: int | None = None  # Star rating 1-5 when known (matches ReviewReplyRequest.rating).
    image: str | None = None  # Path to the customer's photo (photo reviews), see scrable/photos.py.

    def __post_init__(self):
        if self.image is not None and (not isinstance(self.image, str) or not self.image.strip()):
            raise ValueError("image must be a nonempty path string or null")
        if not all(isinstance(v, str) and v.strip() for v in (self.id, self.review, self.app_id)):
            raise ValueError("id, review and app_id must be nonempty strings")
        if self.expert_response is not None and not isinstance(self.expert_response, str):
            raise ValueError("expert_response must be a string or null")
        if self.rating is not None and (type(self.rating) is not int or not 1 <= self.rating <= 5):
            raise ValueError("rating must be an integer from 1 to 5 or null")


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    app_id: str
    source: str


def weighted_score(scores: dict) -> float:
    if set(scores) != set(CRITERIA):
        raise ValueError(f"Expected all four criteria: {CRITERIA}")
    for value in scores.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Scores must be numeric")
        if not math.isfinite(value) or not 1 <= value <= 5:
            raise ValueError("Raw scores must be finite and in [1, 5]")
    return sum(WEIGHTS[k] * (scores[k] - 1) / 4 for k in CRITERIA) / sum(WEIGHTS.values())


@dataclass
class Feedback:
    review_id: str
    review: str
    app_id: str
    prediction: str
    scores: dict[str, float]
    suggestions: list[str]
    rationales: dict[str, str]
    context_ids: list[str] = field(default_factory=list)
    photo_description: str | None = None  # What the vision model said about the photo, if any.

    @property
    def score(self):
        return weighted_score(self.scores)

    def to_dict(self):
        return {**asdict(self), "score": self.score}
