"""Configure original serving entry points without modifying model or prompt logic."""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys
import tempfile


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} before starting this service")
    return value


def customer_config() -> dict:
    device = os.environ.get("CUSTOMER_DEVICE", "cuda")
    if device not in {"cuda", "cpu"}:
        raise ValueError("CUSTOMER_DEVICE must be cuda or cpu")
    aliases = Path(os.environ.get("CUSTOMER_STORE_ALIASES", "/app/store-aliases.json"))
    return {
        "base": {
            "backend": "local",
            "model": os.environ.get("CUSTOMER_MODEL", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"),
            "response_adapter": os.environ.get("CUSTOMER_ADAPTER", "/models/customer"),
            "device": device,
            "cpu_quantization": "int8" if device == "cpu" else None,
            "documents_dir": os.environ.get("CUSTOMER_DOCUMENTS", "/store-docs"),
            "base_prompt_file": "/app/upstream/prompts/stage1_best_prompt.txt",
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "knowledge_size": 0,
            "top_k": 2,
            "output_dir": "/data/customer",
        },
        "store_aliases": json.loads(aliases.read_text(encoding="utf-8")),
        "host": "0.0.0.0",
        "port": 50053,
        "workers": 8,
        "max_waiting": int(os.environ.get("CUSTOMER_MAX_WAITING", "32")),
    }


def command(service: str) -> tuple[str, list[str]]:
    required("AI_INTERNAL_KEY")
    if service == "policy":
        required("SERPAPI_API_KEY")
        # Deliberately call serve_policy.py directly. The research shell script refuses
        # to start when ANY other process uses a GPU, including the customer service.
        return "scripts.serve_policy", [
            "--checkpoint", os.environ.get("POLICY_CHECKPOINT", "/models/policy"),
            "--bind", "0.0.0.0", "--port", "50051", "--pipeline", "mvp",
            "--search-engine", os.environ.get("POLICY_SEARCH_ENGINE", "naver"),
            "--request-timeout", os.environ.get("POLICY_REQUEST_TIMEOUT", "85"),
        ]
    if service == "promotion":
        required("CLOVA_STUDIO_API_KEY")
        required("POSTER_PUBLIC_BASE_URL")
        return "promotion_assistant", [
            "--host", "0.0.0.0", "--port", "8081",
            "--grpc-address", "0.0.0.0:50052", "--env-file", "/dev/null",
        ]
    if service == "management":
        # This deployment intentionally leaves GPU capacity for the two text models.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "chronos_sales.management_grpc", [
            "--address", "0.0.0.0:50054", "--model-path",
            os.environ.get("CHRONOS_MODEL_PATH", "/models/management"),
        ]
    raise ValueError(f"Unknown service: {service}")


def main() -> None:
    service = sys.argv[1] if len(sys.argv) == 2 else ""
    if service == "customer":
        required("AI_INTERNAL_KEY")
        with tempfile.TemporaryDirectory(prefix="scrable-config-") as directory:
            path = Path(directory) / "serve.json"
            path.write_text(json.dumps(customer_config(), ensure_ascii=False), encoding="utf-8")
            sys.argv = ["scrable.serve", "--config", str(path)]
            runpy.run_module("scrable.serve", run_name="__main__")
        return
    module, arguments = command(service)
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
