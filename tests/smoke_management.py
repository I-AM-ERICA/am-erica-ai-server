"""Exercise real Chronos CPU inference over the canonical Spring gRPC contract."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    import grpc
    import torch
    from runtime.generate_proto import generate
    from runtime.healthcheck import check

    with tempfile.TemporaryDirectory(prefix="chronos-grpc-test-") as directory:
        upstream = Path(directory) / "upstream"
        shutil.copytree(ROOT / "services/management/upstream", upstream)
        generate("management", ROOT / "contracts", upstream)
        sys.path.insert(0, str(upstream / "src"))
        from chronos_sales.assistant_forecast import load_chronos_pipeline
        from chronos_sales.management_grpc import create_server
        from chronos_sales._grpc import management_pb2, management_pb2_grpc
        torch.set_num_threads(2)
        model = load_chronos_pipeline(args.model_dir.resolve(), device="cpu")
        server, port = create_server(model, "test-only-internal-key", "127.0.0.1:0")
        server.start()
        target = f"127.0.0.1:{port}"
        try:
            assert check(target, "america.ai.v1.ManagementAssistant", "test-only-internal-key")
            with grpc.insecure_channel(target) as channel:
                stub = management_pb2_grpc.ManagementAssistantStub(channel)
                request = management_pb2.ForecastRequest(horizon_days=3)
                request.store.store_id = "smoke-test-store"
                for index in range(28):
                    request.history.add(date=(date(2025, 1, 1) + timedelta(days=index)).isoformat(), amount=100000 + (index % 7) * 10000)
                try:
                    stub.ForecastSales(request, timeout=10, metadata=(("x-internal-key", "wrong"),))
                    raise AssertionError("Wrong key was accepted")
                except grpc.RpcError as error:
                    assert error.code() == grpc.StatusCode.UNAUTHENTICATED
                response = stub.ForecastSales(request, timeout=90, metadata=(("x-internal-key", "test-only-internal-key"),))
                assert len(response.points) == 3
                for index, point in enumerate(response.points):
                    assert point.date == (date(2025, 1, 29) + timedelta(days=index)).isoformat()
                    assert 0 <= point.lower <= point.predicted <= point.upper
                assert response.model == "chronos-t5-small"
                print("PASS: real CPU model, gRPC health, rejected wrong key, 3 forecast dates and ordered quantiles")
        finally:
            server.stop(0).wait()


if __name__ == "__main__":
    main()
