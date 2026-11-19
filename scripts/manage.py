#!/usr/bin/env python3
"""AM-ERICA AI service deployment CLI (Python 3.10+, standard library only)."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("gpu", "cpu")
PORTS = {"gpu": ("POLICY_PORT", "CUSTOMER_PORT"), "cpu": ("PROMOTION_PORT", "MANAGEMENT_PORT", "POSTER_HTTP_PORT")}


class ConfigError(ValueError):
    pass


def read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ConfigError(f"Missing {path.name}; run: python3 scripts/manage.py init")
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or key in result:
            raise ConfigError(f"{path.name}:{number}: expected a unique KEY=value")
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
            if "'" in value:
                raise ConfigError(f"{path.name}:{number}: embedded quotes are not supported")
        elif any(c in value for c in ('"', "'", "$", "#")):
            raise ConfigError(f"{path.name}:{number}: use a single-quoted literal value; no variable expansion")
        result[key] = value
    return result


def write_private(path: Path, content: str, *, force: bool = False) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if force else os.O_EXCL)
    # O_NOFOLLOW prevents replacing a symlink to some other configuration file.
    flags |= getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(content)


def init(root: Path = ROOT) -> None:
    existing = [read_env(root / f".env.{g}").get("AI_INTERNAL_KEY", "") for g in GROUPS if (root / f".env.{g}").exists()]
    if any(not k or k == "__GENERATE__" for k in existing) or len(set(existing)) > 1:
        raise ConfigError("Existing GPU/CPU internal keys are empty or different; reconcile them first")
    key = existing[0] if existing else secrets.token_hex(32)
    for group in GROUPS:
        path = root / f".env.{group}"
        if not path.exists():
            content = (root / f"deploy/{group}/.env.example").read_text(encoding="utf-8")
            content = content.replace("__ROOT__", str(root.resolve())).replace("__GENERATE__", key)
            write_private(path, content)
            print(f"Created {path.name} (mode 600)")
        else:
            print(f"Kept existing {path.name}")
    (root / "data/models").mkdir(parents=True, exist_ok=True)
    (root / "data/store-docs").mkdir(parents=True, exist_ok=True)
    print("Set API keys, model paths, and real store documents before running up.")


def verify_sources(root: Path = ROOT) -> list[str]:
    manifest = json.loads((root / "sources.lock.json").read_text(encoding="utf-8"))
    errors = []
    for service, source in manifest["sources"].items():
        base = root / f"services/{service}/upstream"
        for relative, expected in source["files"].items():
            path = base / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f"Upstream snapshot changed/missing: {service}/{relative}")
    for relative, expected in manifest["contracts"].items():
        path = root / "contracts" / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"Contract changed/missing: {relative}")
    return errors


def validate(group: str, cfg: dict[str, str]) -> list[str]:
    errors = []
    def require(name):
        if not cfg.get(name) or cfg[name] in {"__GENERATE__", "CHANGE_ME"}:
            errors.append(f"Set {name} in .env.{group}")
    def files(name, names):
        require(name)
        base = Path(cfg.get(name, "/__unset__"))
        if not base.is_absolute():
            errors.append(f"{name} must be an absolute host path")
        for item in names:
            if not (base / item).is_file() or (base / item).stat().st_size == 0:
                errors.append(f"Missing/empty {name}/{item}")
    require("AI_INTERNAL_KEY")
    for name in PORTS[group] + ("STARTUP_TIMEOUT",):
        try:
            value = int(cfg[name])
            if not 1 <= value <= (65535 if name.endswith("PORT") else 86400):
                raise ValueError
        except (ValueError, KeyError):
            errors.append(f"Invalid {name}")
    for name in ("GRPC_BIND_HOST",) + (("POSTER_HTTP_BIND_HOST",) if group == "cpu" else ()):
        try:
            ipaddress.IPv4Address(cfg.get(name, ""))
        except ValueError:
            errors.append(f"{name} must be a host IPv4 address (127.0.0.1 for SSH forwarding)")
    if len({cfg.get(p) for p in PORTS[group]}) != len(PORTS[group]):
        errors.append("Service ports must be distinct")
    if group == "gpu":
        require("SERPAPI_API_KEY")
        files("POLICY_MODEL_DIR", ("config.json", "metadata.json", "actor/adapter_model.safetensors", "tokenizer/tokenizer_config.json"))
        tokenizer = Path(cfg.get("POLICY_MODEL_DIR", "/__unset__")) / "tokenizer"
        if not any((tokenizer / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
            errors.append("Policy checkpoint is missing tokenizer vocabulary")
        files("CUSTOMER_ADAPTER_DIR", ("adapter_config.json", "adapter_model.safetensors"))
        require("CUSTOMER_DOCUMENTS_DIR")
        docs = Path(cfg.get("CUSTOMER_DOCUMENTS_DIR", "/__unset__"))
        if not docs.is_absolute():
            errors.append("CUSTOMER_DOCUMENTS_DIR must be an absolute host path")
        if not docs.is_dir() or not any(p.is_file() and p.suffix.lower() in {".md", ".txt", ".pdf"} and p.stat().st_size for p in docs.glob("*/*")):
            errors.append("Add real store docs under CUSTOMER_DOCUMENTS_DIR/<Spring store ID>/info.md; an empty corpus cannot start Scrable")
        try:
            aliases = Path(cfg["CUSTOMER_STORE_ALIASES_FILE"])
            if not aliases.is_absolute():
                raise ValueError
            mapping = json.loads(aliases.read_text(encoding="utf-8"))
            if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
                raise ValueError
        except (KeyError, OSError, ValueError):
            errors.append("CUSTOMER_STORE_ALIASES_FILE must be an absolute path to a JSON object of string IDs")
        for name in ("POLICY_GPU_ID", "CUSTOMER_GPU_ID"):
            require(name)
        if cfg.get("POLICY_SEARCH_ENGINE") not in {"naver", "google"}:
            errors.append("POLICY_SEARCH_ENGINE must be naver or google")
        for name in ("POLICY_REQUEST_TIMEOUT", "CUSTOMER_MAX_WAITING"):
            try:
                if int(cfg[name]) <= 0:
                    raise ValueError
            except (KeyError, ValueError):
                errors.append(f"{name} must be a positive integer")
        if cfg.get("HF_HUB_OFFLINE") not in {"0", "1"}:
            errors.append("HF_HUB_OFFLINE must be 0 or 1")
    else:
        require("CLOVA_STUDIO_API_KEY")
        files("CHRONOS_MODEL_DIR", ("config.json", "generation_config.json", "model.safetensors"))
        try:
            url = urlsplit(cfg.get("POSTER_PUBLIC_BASE_URL", ""))
            valid_url = url.scheme in {"http", "https"} and bool(url.netloc) and not (url.username or url.query or url.fragment)
        except ValueError:
            valid_url = False
        if not valid_url:
            errors.append("POSTER_PUBLIC_BASE_URL must be the browser-accessible http(s) image base URL")
        try:
            if not 5 <= float(cfg["POSTER_TIMEOUT_SECONDS"]) <= 85:
                raise ValueError
        except (KeyError, ValueError):
            errors.append("POSTER_TIMEOUT_SECONDS must be 5..85")
    return errors


def configs(groups: tuple[str, ...], root: Path = ROOT) -> dict[str, dict[str, str]]:
    return {group: read_env(root / f".env.{group}") for group in groups}


def check(groups: tuple[str, ...], root: Path = ROOT) -> dict[str, dict[str, str]]:
    settings = configs(groups, root)
    errors = verify_sources(root)
    for group, cfg in settings.items():
        errors.extend(validate(group, cfg))
    if len(settings) == 2:
        if settings["gpu"]["AI_INTERNAL_KEY"] != settings["cpu"]["AI_INTERNAL_KEY"]:
            errors.append("GPU and CPU internal keys must match the Spring AI_INTERNAL_KEY")
        values = [cfg.get(name) for group, cfg in settings.items() for name in PORTS[group]]
        if len(set(values)) != len(values):
            errors.append("GPU and CPU ports overlap on this AI host")
    if errors:
        raise ConfigError("\n".join(errors))
    print("Configuration and model files OK (GPU memory, model access, API quotas and network are not tested).")
    return settings


def compose(group: str, cfg: dict[str, str], arguments: list[str], root: Path = ROOT) -> list[str]:
    # Use parsed literals over inherited shell values, so validation and Compose agree.
    env = dict(os.environ)
    env.update(cfg)
    # Status/logs/down/build must remain usable with expired or not-yet-configured keys.
    for key in ("AI_INTERNAL_KEY", "SERPAPI_API_KEY", "CLOVA_STUDIO_API_KEY", "POSTER_PUBLIC_BASE_URL"):
        if not env.get(key):
            env[key] = "not-configured"
    command = ["docker", "compose", "--env-file", str(root / f".env.{group}"), "-f", str(root / f"deploy/{group}/compose.yaml"), *arguments]
    subprocess.run(command, env=env, cwd=root, check=True)
    return command


def import_models(source_root: Path, root: Path = ROOT) -> None:
    """Copy only inference artifacts, never optimizer state or .env files."""
    source_root = source_root.resolve()
    selection = json.loads((root / "services/policy/upstream/serving/model-selection.json").read_text())
    relative = Path(selection["checkpoint"])
    policy = source_root / "Search-R1" / relative
    if not policy.is_dir():
        policy = source_root / "Search-R1/outputs/audit-input" / relative.relative_to("outputs")
    specs = {
        "policy": (policy, ("config.json", "metadata.json", "actor/adapter_model.safetensors", "actor/adapter_config.json", "tokenizer")),
        "customer": (source_root / "Scrable/adapter", ("adapter_config.json", "adapter_model.safetensors")),
        "management": (source_root / "Chronos/models", ("config.json", "generation_config.json", "model.safetensors")),
    }
    for service, (source, names) in specs.items():
        destination = root / "data/models" / service
        if destination.exists():
            print(f"Kept existing data/models/{service}; not overwritten")
            continue
        missing = [name for name in names if not (source / name).exists()]
        if missing:
            raise ConfigError(f"{service}: missing inference artifacts {missing} in {source}; copy the trained model from your server instead")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A failed copy cannot masquerade as a complete model on the next invocation.
        temporary = destination.with_name(destination.name + ".importing-" + secrets.token_hex(4))
        temporary.mkdir()
        try:
            for name in names:
                src, dst = source / name, temporary / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
                else:
                    shutil.copyfile(src, dst)
            (temporary / "import-source.json").write_text(json.dumps({"source": str(source)}, indent=2) + "\n")
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        print(f"Imported data/models/{service}")


def backend_env(settings: dict, gpu_host: str, cpu_host: str) -> str:
    gpu, cpu = settings["gpu"], settings["cpu"]
    if gpu["AI_INTERNAL_KEY"] != cpu["AI_INTERNAL_KEY"]:
        raise ConfigError("GPU and CPU AI_INTERNAL_KEY values differ")
    for host in (gpu_host, cpu_host):
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", host):
            raise ConfigError("Backend host must be an IPv4 address or DNS name without a port")
    key = gpu["AI_INTERNAL_KEY"]
    # Spring .properties files have different quoting/escape rules from Compose dotenv.
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,}", key):
        raise ConfigError("For backend-env, use a shared internal key with 16+ letters/digits/_/- (init generates one)")
    return f"""# Merge into the backend's existing .env. Keep DB/JWT/OAuth settings.
# Hosts below are from the Spring machine's viewpoint; defaults assume SSH tunnels.
AI_INTERNAL_KEY={key}
AI_CUSTOMER_INTERNAL_KEY={key}
AI_CUSTOMER_KEY_FILE=
AI_MOCK_ENABLED=false
AI_POLICY_MOCK_ENABLED=false
AI_CUSTOMER_MOCK_ENABLED=false
AI_MANAGEMENT_MOCK_ENABLED=false
AI_PROMOTION_AUTOSTART=false
AI_MANAGEMENT_AUTOSTART=false
AI_GRPC_TLS=false
AI_GRPC_DEADLINE=90s
AI_CUSTOMER_DEADLINE=300s
AI_POLICY_HOST={gpu_host}
AI_POLICY_PORT={gpu['POLICY_PORT']}
AI_CUSTOMER_HOST={gpu_host}
AI_CUSTOMER_PORT={gpu['CUSTOMER_PORT']}
AI_PROMOTION_HOST={cpu_host}
AI_PROMOTION_PORT={cpu['PROMOTION_PORT']}
AI_MANAGEMENT_HOST={cpu_host}
AI_MANAGEMENT_PORT={cpu['MANAGEMENT_PORT']}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create private env files; never overwrite existing ones")
    commands.add_parser("verify-sources", help="Check upstream snapshot and gRPC contract hashes")
    importer = commands.add_parser("import-models", help="Copy inference weights from the original sibling projects")
    importer.add_argument("--source-root", type=Path, default=ROOT.parent)
    backend = commands.add_parser("backend-env", help="Write .env.backend-ai for merging into Spring settings")
    backend.add_argument("--gpu-host", default="127.0.0.1")
    backend.add_argument("--cpu-host", default="127.0.0.1")
    backend.add_argument("--force", action="store_true")
    for command in ("check", "build", "up", "down", "status", "logs"):
        sub = commands.add_parser(command)
        sub.add_argument("group", choices=(*GROUPS, "all"))
    args = parser.parse_args()
    try:
        if args.command == "init":
            init()
        elif args.command == "verify-sources":
            errors = verify_sources()
            if errors:
                raise ConfigError("\n".join(errors))
            print("All imported source and contract hashes match.")
        elif args.command == "import-models":
            import_models(args.source_root)
        elif args.command == "backend-env":
            text = backend_env(configs(GROUPS), args.gpu_host, args.cpu_host)
            write_private(ROOT / ".env.backend-ai", text, force=args.force)
            print("Wrote .env.backend-ai (mode 600). Merge into Spring's existing .env; do not replace it wholesale.")
        else:
            groups = GROUPS if args.group == "all" else (args.group,)
            settings = check(groups) if args.command in {"check", "up"} else configs(groups)
            if args.command == "check":
                return 0
            if not shutil.which("docker"):
                raise ConfigError("Docker is not installed. On the AI server install Docker Engine + Compose; GPU services also need NVIDIA Container Toolkit.")
            subprocess.run(["docker", "compose", "version"], check=True)
            if args.command == "up":
                # Preflight every group before starting any container.
                for group in groups:
                    compose(group, settings[group], ["config", "--quiet"])
            for group in groups:
                cfg = settings[group]
                action = {
                    "build": ["build"],
                    "up": ["up", "--detach", "--build", "--wait", "--wait-timeout", cfg.get("STARTUP_TIMEOUT", "1800")],
                    "down": ["down"],  # Deliberately never deletes model/data volumes.
                    "status": ["ps"],
                    "logs": ["logs", "--tail", "100"],
                }[args.command]
                compose(group, cfg, action)
            if args.command == "up":
                print("Selected services are healthy. Configure Spring endpoints/tunnels using .env.backend-ai.")
        return 0
    except (ConfigError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
