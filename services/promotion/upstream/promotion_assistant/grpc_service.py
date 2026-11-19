from concurrent.futures import ThreadPoolExecutor
import base64
import hmac
import json
import logging
import time
import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from america.ai.v1 import promotion_pb2 as pb, promotion_pb2_grpc as rpc
from .clova import ProviderError
from .models import Brief, Revision
from .storage import NotFound, Conflict


def brief_from_proto(request):
    data = {
        "store": {key: getattr(request.store, key) for key in ("store_id", "name", "category", "region", "address")},
        "product": request.product, "message": request.message,
        "tone": request.tone or "따뜻하고 친근하게", "channel": request.channel or "instagram",
        "period": request.period, "contact": request.contact,
        "background_mode": request.background_mode or "auto",
        "hide_footer": request.hide_footer,
    }
    for key in ("price", "original_price"):
        if request.HasField(key):
            data[key] = getattr(request, key)
    for key in ("product_image", "logo", "style_reference"):
        if request.HasField(key):
            value = getattr(request, key)
            data[key] = {"base64": base64.b64encode(value.data).decode(), "media_type": value.media_type}
    return Brief.model_validate(data)


def draft_message(record):
    return pb.PosterDraft(
        id=record["id"], version=record["version"], mode=record["mode"],
        result=pb.PosterResult(**record["caption"], image_url=record["image_url"]),
        design_json=json.dumps(record["design"], ensure_ascii=False),
        width=record["width"], height=record["height"],
        image_processing_json=json.dumps(record.get("image_processing", {}), ensure_ascii=False),
    )


class PromotionService(rpc.PromotionAssistantServicer):
    def __init__(self, engine):
        self.engine = engine
        self.key = engine.settings.internal_key.encode()

    def invoke(self, context, operation):
        supplied = dict(context.invocation_metadata()).get("x-internal-key", "")
        if not isinstance(supplied, str) or not hmac.compare_digest(supplied.encode(), self.key):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid internal key")
        remaining = context.time_remaining()
        deadline = time.monotonic() + min(remaining if remaining is not None else 85, self.engine.settings.timeout)
        try:
            return operation({"deadline": deadline, "cancelled": lambda: not context.is_active()})
        except NotFound as error:
            context.abort(grpc.StatusCode.NOT_FOUND, str(error))
        except Conflict as error:
            context.abort(grpc.StatusCode.ABORTED, str(error))
        except ProviderError as error:
            code = {
                "DEADLINE": grpc.StatusCode.DEADLINE_EXCEEDED,
                "BUSY": grpc.StatusCode.RESOURCE_EXHAUSTED,
                "RATE_LIMIT": grpc.StatusCode.RESOURCE_EXHAUSTED,
                "FACT_EDIT": grpc.StatusCode.INVALID_ARGUMENT,
                "DEMO": grpc.StatusCode.FAILED_PRECONDITION,
                "AUTH": grpc.StatusCode.FAILED_PRECONDITION,
            }.get(error.code, grpc.StatusCode.UNAVAILABLE)
            context.abort(code, str(error))
        except ValueError:
            # Pydantic error strings can contain input images; do not echo them.
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "입력 정보나 이미지, 문구 길이를 확인해 주세요.")
        except Exception as error:
            logging.error("Poster request failed: %s", type(error).__name__)
            context.abort(grpc.StatusCode.INTERNAL, "포스터 처리 중 오류가 발생했습니다.")

    def CreatePoster(self, request, context):
        return self.invoke(context, lambda budget: pb.PosterResult(**self.engine.create_legacy(
            brief_from_proto(request), request.generate_image, **budget)))

    def CreateDrafts(self, request, context):
        def run(budget):
            result = self.engine.create(brief_from_proto(request), **budget)
            return pb.PosterDrafts(drafts=[draft_message(d) for d in result["drafts"]], mode=result["mode"])
        return self.invoke(context, run)

    def RevisePoster(self, request, context):
        return self.invoke(context, lambda budget: draft_message(self.engine.revise(
            request.draft_id, Revision(store_id=request.store_id, instruction=request.instruction,
                                      expected_version=request.expected_version), **budget)))

    def GetPoster(self, request, context):
        return self.invoke(context, lambda _: draft_message(self.engine.get(request.draft_id, request.store_id)))


def build_server(engine, address="127.0.0.1:50052"):
    server = grpc.server(ThreadPoolExecutor(max_workers=4), options=[
        ("grpc.max_receive_message_length", 17 * 1024 * 1024),
        ("grpc.max_send_message_length", 1024 * 1024)])
    rpc.add_PromotionAssistantServicer_to_server(PromotionService(engine), server)
    status = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(status, server)
    status.set("", health_pb2.HealthCheckResponse.SERVING)
    status.set("america.ai.v1.PromotionAssistant", health_pb2.HealthCheckResponse.SERVING)
    port = server.add_insecure_port(address)
    if not port:
        raise RuntimeError("gRPC 포트를 열 수 없습니다.")
    return server, port
