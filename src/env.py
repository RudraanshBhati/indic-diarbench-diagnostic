"""Environment/credential loading.

Reads `.env` at the repo root. The token value is never printed or logged — only
its presence and the variable name it came from.
"""

from __future__ import annotations

import os

from constants import REPO_ROOT

# Accept the common spellings so a differently-cased key in .env still works.
_TOKEN_KEYS = ["HF_TOKEN", "HF_Token", "hf_token", "HUGGINGFACE_TOKEN",
               "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"]


def load_env() -> None:
    """Load .env into os.environ without overriding an already-set variable."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)


def get_hf_token(required: bool = False) -> str | None:
    """Return the HF token, or None. Never logs the value."""
    load_env()
    for key in _TOKEN_KEYS:
        val = os.environ.get(key)
        if val and val.strip():
            token = val.strip()
            # Normalize so huggingface_hub picks it up automatically downstream.
            os.environ["HF_TOKEN"] = token
            return token
    if required:
        raise RuntimeError(
            f"No Hugging Face token found. Set one of {_TOKEN_KEYS} in .env"
        )
    return None


def token_status() -> str:
    """Human-readable status line that reveals nothing about the token itself."""
    load_env()
    for key in _TOKEN_KEYS:
        val = os.environ.get(key)
        if val and val.strip():
            return f"HF token: found (from {key}, {len(val.strip())} chars, value not shown)"
    return "HF token: not found"
