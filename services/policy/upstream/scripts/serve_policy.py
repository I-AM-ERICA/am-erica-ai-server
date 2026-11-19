"""Serve a saved Actor using SerpAPI and the existing Spring gRPC contract."""
import argparse
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--request-timeout", type=float, default=85)
    parser.add_argument("--search-engine", choices=("google", "naver"), default="naver")
    parser.add_argument("--search-timeout", type=float, default=15)
    parser.add_argument("--max-new-tokens", type=int, help="Serving-only generation limit; default: checkpoint setting")
    parser.add_argument("--pipeline", choices=("mvp", "agent"), default="mvp")
    parser.add_argument("--bind", default="127.0.0.1", help="Use a private interface or SSH tunnel for Spring")
    args = parser.parse_args()
    from policy_r1.runtime_env import load_runtime_env
    load_runtime_env()
    if not 1 <= args.port <= 65535 or args.request_timeout <= 0:
        parser.error("Invalid port/request timeout")
    internal_key = os.environ.get("AI_INTERNAL_KEY", "")
    if not internal_key:
        parser.error("Set AI_INTERNAL_KEY on the server; use the same value in Spring")
    from policy_r1.web_search import SerpAPIRetriever
    retriever = SerpAPIRetriever(engine=args.search_engine, timeout=args.search_timeout, require_page_evidence=True)
    from policy_r1.inference import load_actor, PolicyInference, with_generation_limit
    from policy_r1.grpc_service import build_server
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("Loading Actor checkpoint:", Path(args.checkpoint).resolve(), flush=True)
    tokenizer, actor, config = load_actor(args.checkpoint)
    config = with_generation_limit(config, args.max_new_tokens)
    if args.pipeline == "mvp":
        from policy_r1.mvp import MVPPolicyInference
        engine = MVPPolicyInference(tokenizer, actor, config, retriever)
    else:
        engine = PolicyInference(tokenizer, actor, config, retriever)
    server, _ = build_server(engine, internal_key, f"{args.bind}:{args.port}", args.request_timeout)
    server.start()
    print(f"POLICY SERVER READY: {args.bind}:{args.port}; pipeline={args.pipeline}; engine={args.search_engine}; evidence=read HTML pages; max_searches<=2; max_new_tokens={config.max_new_tokens}", flush=True)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=3).wait()


if __name__ == "__main__":
    main()
