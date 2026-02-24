# -*- coding: utf-8 -*-
"""Load environment variables from .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (codeCleaning folder)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

# API keys and secrets (never commit .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Direct OpenAI key for endpoints not supported by the gateway (e.g. embeddings).
# Falls back to OPENAI_API_KEY if not set separately.
OPENAI_DIRECT_API_KEY = os.getenv("OPENAI_DIRECT_API_KEY", "") or OPENAI_API_KEY
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# OpenAI-compatible gateway (e.g. Smallcase AI Gateway)
# When set, all ChatOpenAI and OpenAIEmbeddings calls use this base URL.
_openai_base = (os.getenv("OPENAI_API_BASE") or "https://ai-gateway.smallcase.com/").strip()
OPENAI_API_BASE = (_openai_base.rstrip("/") + "/") if _openai_base else ""

# Optional: default model names (DEFAULT_OPENAI_MODEL is the "model" param sent to the gateway)
DEFAULT_OPENAI_MODEL = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_HF_MODEL = os.getenv("DEFAULT_HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
