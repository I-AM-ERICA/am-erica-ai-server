from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .assistant_storage import RevenueRecord, validate_forecast_history


FORECAST_DAYS = 14
NUM_SAMPLES = 20


@dataclass(frozen=True)
class ForecastPoint:
    base_date: date
    forecast_date: date
    days_ahead: int
    pred_q10: int
    pred_q50: int
    pred_q90: int
    currency: str = "KRW"

    def as_api_dict(self) -> dict[str, str | int]:
        result = asdict(self)
        result["base_date"] = self.base_date.isoformat()
        result["forecast_date"] = self.forecast_date.isoformat()
        return result


def model_identifier(model_path: Path | str) -> str:
    path = Path(model_path).expanduser().resolve()
    model_file = path / "model.safetensors"
    if model_file.exists():
        stat = model_file.stat()
        return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
    return str(path)


def forecast_cache_key(
    records: Sequence[RevenueRecord], model_path: Path | str
) -> str:
    payload = {
        "model": model_identifier(model_path),
        "history": [(record.date.isoformat(), record.revenue) for record in records],
        "prediction_length": FORECAST_DAYS,
        "num_samples": NUM_SAMPLES,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve_inference_device(device: str = "auto") -> tuple[str, torch.dtype]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 사용할 수 없습니다.")
    return device, torch.bfloat16 if device == "cuda" else torch.float32


def load_chronos_pipeline(model_path: Path | str, device: str = "auto") -> Any:
    from chronos import ChronosPipeline

    resolved_device, dtype = resolve_inference_device(device)
    return ChronosPipeline.from_pretrained(
        str(Path(model_path).expanduser().resolve()),
        device_map=resolved_device,
        dtype=dtype,
    )


def _to_numpy_samples(raw: Any, prediction_length: int = FORECAST_DAYS) -> np.ndarray:
    if isinstance(raw, torch.Tensor):
        samples = raw.detach().cpu().float().numpy()
    else:
        samples = np.asarray(raw, dtype=np.float32)
    if samples.ndim == 3 and samples.shape[0] == 1:
        samples = samples[0]
    if samples.ndim != 2:
        raise RuntimeError(
            f"Chronos 예측 크기가 예상과 다릅니다: {samples.shape}"
        )
    if samples.shape[1] != prediction_length:
        raise RuntimeError(
            f"Chronos가 {prediction_length}일이 아닌 {samples.shape[1]}일을 반환했습니다."
        )
    if not np.isfinite(samples).all():
        raise RuntimeError("Chronos 예측에 유효하지 않은 숫자가 포함되어 있습니다.")
    return samples


def forecast_with_pipeline(
    pipeline: Any,
    records: Sequence[RevenueRecord],
    *,
    seed: int = 42,
    prediction_length: int = FORECAST_DAYS,
    min_history_days: int = 28,
    max_context_days: int = 28,
) -> list[ForecastPoint]:
    if not 1 <= prediction_length <= 30:
        raise ValueError("예측 기간은 1~30일이어야 합니다.")
    history = validate_forecast_history(records, min_days=min_history_days, max_context=max_context_days)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    context = torch.tensor(
        [float(record.revenue) for record in history], dtype=torch.float32
    )
    raw = pipeline.predict(
        [context], prediction_length=prediction_length, num_samples=NUM_SAMPLES
    )
    samples = np.maximum(_to_numpy_samples(raw, prediction_length), 0.0)
    quantiles = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)
    base_date = history[-1].date
    points: list[ForecastPoint] = []
    for index in range(prediction_length):
        points.append(
            ForecastPoint(
                base_date=base_date,
                forecast_date=base_date + timedelta(days=index + 1),
                days_ahead=index + 1,
                pred_q10=max(0, round(float(quantiles[0, index]))),
                pred_q50=max(0, round(float(quantiles[1, index]))),
                pred_q90=max(0, round(float(quantiles[2, index]))),
            )
        )
    return points


def select_forecast(
    forecasts: Sequence[ForecastPoint], selected_date: date
) -> ForecastPoint:
    for point in forecasts:
        if point.forecast_date == selected_date:
            return point
    raise ValueError(
        f"선택한 날짜가 {FORECAST_DAYS}일 예측 범위에 없습니다."
    )
