"""All the knobs in one place. Override any of them in your .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
DB_PATH = Path(os.getenv("METRICS_DB", ROOT / "data" / "metrics.db"))

# --- Models -----------------------------------------------------------------
# Small = cheap and fast, for simple lookups. Large = smart and pricier, for
# questions that need reasoning. The "baseline" sends everything to LARGE.
SMALL_MODEL = os.getenv("SMALL_MODEL", "claude-haiku-5-5")
LARGE_MODEL = os.getenv("LARGE_MODEL", "claude-opus-5-5")

# USD per 1 million tokens: (input, output). Check https://claude.com/pricing
# if you change models.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

# How hard each model "thinks". Lower effort = fewer tokens = cheaper/faster.
EFFORT = {
    SMALL_MODEL: os.getenv("SMALL_EFFORT", "low"),
    LARGE_MODEL: os.getenv("LARGE_EFFORT", "medium"),
}
MAX_OUTPUT_TOKENS = 4000

# --- Retrieval --------------------------------------------------------------
TOP_K = 4                # how many document chunks to give the model
CHUNK_MAX_CHARS = 900    # rough chunk size

# "fastembed" = real semantic embeddings (BAAI/bge-small-en-v1.5, runs locally,
# free). "hash" = tiny keyword-based fallback for offline testing only.
EMBEDDER = os.getenv("EMBEDDER", "fastembed")

# --- Semantic cache ---------------------------------------------------------
# Two questions with cosine similarity >= this are treated as "the same
# question". Higher = safer but fewer hits. Tune it with benchmark.py.
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.90"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(24 * 3600)))
CACHE_MAX_ENTRIES = 5000
# Embeddings are bad at telling "trash on the Plus plan" from "trash on the Free
# plan" (similarity ~0.95, different answers). The cache refuses a match when
# the two questions mention different key terms from this list, or different
# numbers.
CACHE_KEY_TERMS = ["free", "plus", "business", "student", "admin",
                   "windows", "mac", "macos", "linux", "ubuntu", "ios", "android",
                   "monthly", "annual", "yearly"]

# --- Router -----------------------------------------------------------------
# A question with a complexity score >= this goes to the large model.
ROUTER_HARD_THRESHOLD = int(os.getenv("ROUTER_HARD_THRESHOLD", "2"))


def price_of(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * inp + output_tokens * out) / 1_000_000
