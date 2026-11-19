"""Standard gRPC health probe: no inference, external API calls, or paid requests."""
from __future__ import annotations

import argparse
import os
import sys

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


def check(target: str, service: str, key: str, timeout: float = 5) -> bool:
    with grpc.insecure_channel(target) as channel:
        response = health_pb2_grpc.HealthStub(channel).Check(
            health_pb2.HealthCheckRequest(service=service),
            metadata=(("x-internal-key", key),), timeout=timeout,
        )
        return response.status == health_pb2.HealthCheckResponse.SERVING


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target")
    parser.add_argument("service")
    args = parser.parse_args()
    try:
        if check(args.target, args.service, os.environ.get("AI_INTERNAL_KEY", "")):
            return 0
        print("Service is not SERVING", file=sys.stderr)
    except grpc.RpcError as error:
        print(f"Health check failed: {error.code().name}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
