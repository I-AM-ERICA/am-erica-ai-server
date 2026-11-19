"""Read only serving credentials from .env, without executing shell code."""
import os
from pathlib import Path
import shlex


def load_runtime_env(path=".env"):
    file = Path(path)
    if not file.is_file():
        return
    for number, line in enumerate(file.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or key not in {"SERPAPI_API_KEY", "AI_INTERNAL_KEY"}:
            continue
        try:
            parts = shlex.split(value, comments=True)
        except ValueError:
            raise ValueError(f"Invalid credential quoting in {file.name}:{number}") from None
        if len(parts) > 1:
            raise ValueError(f"Credential must be a single value in {file.name}:{number}")
        os.environ.setdefault(key, parts[0] if parts else "")
