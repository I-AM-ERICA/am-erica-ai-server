"""Stateless, authenticated AM-ERICA adapter for the real Chronos checkpoint."""
from __future__ import annotations

import argparse
from concurrent import futures
from datetime import date, datetime
import hmac
import logging
import os
from pathlib import Path
import signal
import threading
from zoneinfo import ZoneInfo

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
import torch

from ._grpc import management_pb2, management_pb2_grpc
from .assistant_forecast import forecast_with_pipeline, load_chronos_pipeline
from .assistant_storage import RevenueDataError, RevenueRecord, validate_forecast_history

LOG = logging.getLogger(__name__)
MODEL_NAME = "chronos-t5-small"
MAX_HISTORY = 512
MAX_AMOUNT = 9_007_199_254_740_991


def parse_history(request) -> list[RevenueRecord]:
    if not 1 <= request.horizon_days <= 30:
        raise RevenueDataError("예측 기간은 1~30일이어야 합니다.")
    if not 1 <= len(request.history) <= MAX_HISTORY:
        raise RevenueDataError("매출은 1~512일을 입력해 주세요.")
    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    records = []
    for point in request.history:
        try:
            day = date.fromisoformat(point.date)
        except ValueError as exc:
            raise RevenueDataError("날짜 형식은 YYYY-MM-DD여야 합니다.") from exc
        if day.isoformat() != point.date or day > today:
            raise RevenueDataError("실제 매출 날짜는 오늘까지 입력할 수 있습니다.")
        if not 0 <= point.amount <= MAX_AMOUNT:
            raise RevenueDataError("매출은 0 이상의 원 단위 정수여야 합니다.")
        records.append(RevenueRecord(request.store.store_id, day, point.amount, "request", ""))
    return validate_forecast_history(records, min_days=1, max_context=MAX_HISTORY)


class ManagementService(management_pb2_grpc.ManagementAssistantServicer):
    def __init__(self, pipeline, internal_key: str):
        self.pipeline = pipeline
        self.internal_key = internal_key
        # Sampling sets the torch seed; serialize requests for deterministic results.
        self.lock = threading.Lock()

    def ForecastSales(self, request, context):
        provided = dict(context.invocation_metadata()).get("x-internal-key", "")
        if not hmac.compare_digest(provided, self.internal_key):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid internal key")
        try:
            records = parse_history(request)
        except RevenueDataError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        while not self.lock.acquire(timeout=0.1):
            if not context.is_active():
                context.abort(grpc.StatusCode.CANCELLED, "request cancelled")
        try:
            if not context.is_active():
                context.abort(grpc.StatusCode.CANCELLED, "request cancelled")
            points = forecast_with_pipeline(
                self.pipeline, records, prediction_length=request.horizon_days,
                min_history_days=1, max_context_days=MAX_HISTORY,
            )
        except (RuntimeError, ValueError):
            LOG.exception("Chronos inference failed")
            context.abort(grpc.StatusCode.INTERNAL, "Chronos inference failed")
        finally:
            self.lock.release()
        if len(points) == 1:
            summary = f"{points[0].forecast_date.isoformat()} 예상 매출은 {points[0].pred_q50:,}원입니다."
        else:
            summary = f"{points[0].forecast_date.isoformat()}부터 {len(points)}일간 일별 예상 매출을 계산했습니다."
        summary += f" 실제 매출 {len(records)}일을 사용했습니다."
        if len(records) < 28:
            summary += " 입력 이력이 짧아 매장의 반복 패턴을 충분히 반영하지 못할 수 있습니다."
        return management_pb2.ForecastResult(
            points=[management_pb2.ForecastPoint(
                date=p.forecast_date.isoformat(), predicted=p.pred_q50,
                lower=p.pred_q10, upper=p.pred_q90,
            ) for p in points], summary=summary, model=MODEL_NAME,
        )


def create_server(pipeline, internal_key: str, address: str):
    if not internal_key:
        raise ValueError("AI_INTERNAL_KEY is required")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), maximum_concurrent_rpcs=8)
    management_pb2_grpc.add_ManagementAssistantServicer_to_server(
        ManagementService(pipeline, internal_key), server,
    )
    status = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(status, server)
    status.set("", health_pb2.HealthCheckResponse.SERVING)
    status.set("america.ai.v1.ManagementAssistant", health_pb2.HealthCheckResponse.SERVING)
    port = server.add_insecure_port(address)
    if not port:
        raise RuntimeError("Could not bind management gRPC address")
    return server, port


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="127.0.0.1:50054")
    parser.add_argument("--model-path", default=os.getenv("CHRONOS_MODEL_PATH", "models"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    key = os.getenv("AI_INTERNAL_KEY", "")
    if not key:
        parser.error("AI_INTERNAL_KEY is required")
    model = Path(args.model_path).expanduser().resolve()
    for name in ("config.json", "generation_config.json", "model.safetensors"):
        if not (model / name).is_file():
            parser.error(f"Missing model file: {model / name}")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    pipeline = load_chronos_pipeline(model)
    server, _ = create_server(pipeline, key, args.address)
    server.start()
    LOG.info("Chronos ready at %s (model loaded)", args.address)
    signal.signal(signal.SIGTERM, lambda *_: server.stop(3))
    signal.signal(signal.SIGINT, lambda *_: server.stop(3))
    server.wait_for_termination()


if __name__ == "__main__":
    main()
