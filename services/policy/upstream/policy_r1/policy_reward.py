"""OFFLINE calibration only: reviewed policy outcomes, not a live PPO judge.

The existing trainer never imports this module. Semantic grades require an
evidence-aware reviewer; string containment/domain presence cannot supply them.
"""
from .policy_sft import fingerprint


def reviewed_policy_reward(candidate, review):
    if (review.get("candidate_sha256") != fingerprint(candidate)
            or not review.get("reviewer") or not review.get("notes")):
        raise ValueError("Missing/stale source-grounded review")
    fields = ("accuracy", "coverage", "grounding")
    if any(type(review.get(k)) is not int or review[k] not in (0, 1, 2) for k in fields):
        raise ValueError("All semantic grades must be explicit integers 0..2")
    for flag in ("critical_error", "unsupported_claim", "justified_abstention"):
        if type(review.get(flag)) is not bool:
            raise ValueError("Missing explicit semantic flag")
    if (not candidate.get("answer", "").strip() or candidate.get("finish_reason") != "answer"
            or review["critical_error"] or review["unsupported_claim"] or review["accuracy"] == 0):
        return 0.0
    if review["justified_abstention"]:
        # Safe failure is distinct from completing the task; cannot beat a fully
        # supported complete answer by always refusing to answer.
        return min(0.25, 0.125 * review["coverage"])
    if review["grounding"] == 0:
        return 0.0
    return (0.5 * review["accuracy"] + 0.25 * review["coverage"] + 0.25 * review["grounding"]) / 2
