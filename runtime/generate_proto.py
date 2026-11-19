"""Build service bindings from the shared Spring contract using each image's runtime."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from grpc_tools import protoc


def generate(service: str, contracts: Path, upstream: Path) -> None:
    name = {"policy": "policy", "customer": "customer", "promotion": "promotion", "management": "management"}[service]
    protos = [f"america/ai/v1/{p}.proto" for p in ("common", name)]
    with tempfile.TemporaryDirectory() as temp:
        code = protoc.main(["protoc", f"-I{contracts}", f"--python_out={temp}", f"--grpc_python_out={temp}", *protos])
        if code:
            raise RuntimeError("gRPC code generation failed")
        generated = Path(temp) / "america/ai/v1"
        if service == "management":
            destination = upstream / "src/chronos_sales/_grpc"
        elif service == "customer":
            destination = upstream / "scrable/_grpc_gen/america/ai/v1"
        else:
            destination = upstream / "america/ai/v1"
        destination.mkdir(parents=True, exist_ok=True)
        for path in generated.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            if service == "management":
                text = text.replace("from america.ai.v1 import ", "from . import ")
            (destination / path.name).write_text(text, encoding="utf-8")
        if service == "management":
            (destination / "__init__.py").touch()
        if service == "customer":
            # Scrable checks these timestamps before generating bindings at runtime.
            for proto in protos:
                target = upstream / "scrable/proto" / proto
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((contracts / proto).read_bytes())
            for path in destination.glob("*.py"):
                path.touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=("policy", "customer", "promotion", "management"))
    parser.add_argument("--contracts", type=Path, default=Path("/app/contracts"))
    parser.add_argument("--upstream", type=Path, default=Path("/app/upstream"))
    args = parser.parse_args()
    generate(args.service, args.contracts, args.upstream)
