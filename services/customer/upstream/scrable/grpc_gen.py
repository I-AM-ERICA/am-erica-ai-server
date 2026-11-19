"""Generate and import the gRPC code for the backend's AI contract (america.ai.v1).

The .proto files in scrable/proto are copied unchanged from am-erica-backend
(src/main/proto/america/ai/v1). The Python code is generated here on first use, with the
installed grpcio-tools, so it always matches the installed protobuf runtime. Generated files go
to scrable/_grpc_gen (gitignored) and are rebuilt when a .proto file is newer.
"""

import sys
from pathlib import Path

PROTO_ROOT = Path(__file__).resolve().parent / "proto"
OUT_DIR = Path(__file__).resolve().parent / "_grpc_gen"
PROTOS = ("america/ai/v1/common.proto", "america/ai/v1/customer.proto")


def _stale():
    for proto in PROTOS:
        stem = OUT_DIR / proto.replace(".proto", "_pb2.py")
        if not stem.exists() or stem.stat().st_mtime < (PROTO_ROOT / proto).stat().st_mtime:
            return True
    return False


def ensure_generated():
    """Generate the *_pb2 modules if needed and put them on sys.path. Returns the output folder."""
    if _stale():
        try:
            from grpc_tools import protoc
        except ImportError as exc:
            raise RuntimeError("grpcio-tools is required: pip install -e '.[serve]'") from exc
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        args = [
            "protoc",
            f"-I{PROTO_ROOT}",
            f"--python_out={OUT_DIR}",
            f"--grpc_python_out={OUT_DIR}",
            *PROTOS,
        ]
        if protoc.main(args) != 0:
            raise RuntimeError(f"protoc failed for {PROTOS}")
    if str(OUT_DIR) not in sys.path:
        sys.path.insert(0, str(OUT_DIR))
    return OUT_DIR


def customer_modules():
    """(customer_pb2, customer_pb2_grpc), generated on first call."""
    ensure_generated()
    from america.ai.v1 import customer_pb2, customer_pb2_grpc

    return customer_pb2, customer_pb2_grpc
