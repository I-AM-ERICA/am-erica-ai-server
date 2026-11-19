from dataclasses import dataclass
from pathlib import Path

from .io import write_json
from .pipeline import average_score, identify_ir
from .prompts import BASE_PROMPT


@dataclass
class OptimizationResult:
    best_prompt: str
    final_prompt: str
    best_iteration: int
    stop_reason: str
    history: list[dict]


class PromptOptimizer:
    def __init__(self, scorer, prompt_generator, config):
        self.scorer = scorer
        self.prompt_generator = prompt_generator
        self.config = config

    def run(self, train, validation, base_prompt=BASE_PROMPT):
        if not train or not validation:
            raise ValueError("Both train and validation reviews are required")
        if {r.id for r in train} & {r.id for r in validation}:
            raise ValueError("Train/validation overlap")
        prompt = base_prompt
        history, best_score, best_iteration, best_prompt = [], -1.0, 0, prompt
        feedback = None
        stop_reason = "max_iterations"
        for iteration in range(self.config.max_iterations + 1):
            selected = []
            if iteration:
                selected = identify_ir(
                    feedback,
                    self.config.lowest_fraction,
                    self.config.random_fraction,
                    self.config.seed + iteration,
                )
                prompt = self.prompt_generator(prompt, selected)
            # Every revision reevaluates all train reviews, not only the selected cases.
            feedback = self.scorer(prompt, train, description=f"Iteration {iteration}: train")
            valid_feedback = self.scorer(prompt, validation, description=f"Iteration {iteration}: validation")
            train_score, valid_score = average_score(feedback), average_score(valid_feedback)
            record = {
                "iteration": iteration,
                "prompt": prompt,
                "train_score": train_score,
                "validation_score": valid_score,
                "selected_ids": [f.review_id for f in selected],
                "train_feedback": [f.to_dict() for f in feedback],
                "validation_feedback": [f.to_dict() for f in valid_feedback],
            }
            history.append(record)
            if valid_score > best_score:  # Stable ties retain earlier/simpler prompt.
                best_score, best_iteration, best_prompt = valid_score, iteration, prompt
            directory = Path(self.config.output_dir)
            write_json(directory / f"iteration_{iteration:03d}.json", record)
            write_json(directory / "history.json", history)
            (directory / "best_prompt.txt").write_text(best_prompt, encoding="utf-8")
            print(f"Iteration {iteration}: train={train_score:.3f}, validation={valid_score:.3f}", flush=True)
            if train_score >= self.config.threshold:
                stop_reason = "threshold"
                break
        (Path(self.config.output_dir) / "final_prompt.txt").write_text(prompt, encoding="utf-8")
        return OptimizationResult(best_prompt, prompt, best_iteration, stop_reason, history)
