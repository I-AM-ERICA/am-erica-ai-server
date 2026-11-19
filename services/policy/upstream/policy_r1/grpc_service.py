"""Implements am-erica-backend's existing PolicyAssistant contract."""
from concurrent.futures import ThreadPoolExecutor
import hmac
import logging
import threading
import time

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from america.ai.v1 import common_pb2, policy_pb2, policy_pb2_grpc
from .inference import InferenceFailure
from .rollout import RolloutCancelled
from .web_search import SearchUnavailable


class PolicyService(policy_pb2_grpc.PolicyAssistantServicer):
    def __init__(self, engine, internal_key, request_timeout=85):
        if not internal_key or request_timeout <= 0:
            raise ValueError("Internal key and positive request timeout required")
        self.engine, self._key, self.timeout = engine, internal_key.encode(), request_timeout
        self._gpu_lock = threading.Lock()

    def _answer(self, request, context):
        supplied = dict(context.invocation_metadata()).get("x-internal-key", "")
        if not isinstance(supplied, str) or not hmac.compare_digest(supplied.encode(), self._key):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid internal key")
        if not request.question.strip() or len(request.question) > 20000 or not 0 <= request.top_k <= 5:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid question or top_k (1..5; 0 uses 3)")
        if not self._gpu_lock.acquire(blocking=False):
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "GPU is busy; retry shortly")
        started = time.monotonic()
        try:
            remaining = context.time_remaining()
            budget = self.timeout if remaining is None else min(self.timeout, remaining)
            result = self.engine.answer(request.question, region=request.store.region,
                                        category=request.store.category, top_k=request.top_k or 3,
                                        cancelled=lambda: not context.is_active(), deadline=started + budget)
            logging.info("Policy request completed in %.2fs; searches=%s", time.monotonic() - started,
                         result.get("searches", "unknown"))
            return policy_pb2.PolicyAnswer(answer=result["answer"], sources=[
                common_pb2.Source(title=s["title"], url=s["url"], snippet=s["snippet"], score=s["score"])
                for s in result["sources"]])
        except RolloutCancelled:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "Inference cancelled or deadline exceeded")
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except SearchUnavailable as error:
            # Only enum-like codes and validated search IDs cross the RPC boundary.
            detail = f"Web search unavailable [{error.code}]"
            if error.search_id:
                detail += f"; search_id={error.search_id}"
            logging.warning("%s; request elapsed=%.2fs", detail, time.monotonic() - started)
            status = grpc.StatusCode.DEADLINE_EXCEEDED if error.code == "DEADLINE" else grpc.StatusCode.UNAVAILABLE
            context.abort(status, detail)
        except InferenceFailure as error:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        except Exception as error:
            logging.error("Policy inference failed (%s)", type(error).__name__)
            context.abort(grpc.StatusCode.INTERNAL, "Model inference failed; inspect the server")
        finally:
            self._gpu_lock.release()

    def Ask(self, request, context):
        return self._answer(request, context)

    def AskStream(self, request, context):
        answer = self._answer(request, context)
        # Protocol-compatible chunks after the search/answer loop finishes.
        # These are NOT live token-generation events or hidden reasoning.
        for start in range(0, len(answer.answer), 80):
            if not context.is_active():
                return
            yield policy_pb2.PolicyAnswerChunk(delta=answer.answer[start:start + 80])
        yield policy_pb2.PolicyAnswerChunk(sources=answer.sources, done=True)


def build_server(engine, internal_key, address="127.0.0.1:50051", request_timeout=85):
    server = grpc.server(ThreadPoolExecutor(max_workers=4), options=[
        ("grpc.max_receive_message_length", 128 * 1024), ("grpc.max_send_message_length", 1024 * 1024)])
    policy_pb2_grpc.add_PolicyAssistantServicer_to_server(PolicyService(engine, internal_key, request_timeout), server)
    status = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(status, server)
    status.set("", health_pb2.HealthCheckResponse.SERVING)
    status.set("america.ai.v1.PolicyAssistant", health_pb2.HealthCheckResponse.SERVING)
    port = server.add_insecure_port(address)
    if port == 0:
        raise RuntimeError("Could not bind gRPC port")
    return server, port
