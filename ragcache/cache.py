"""Semantic cache: if a new question means the same thing as one we already
answered, return the stored answer instead of calling the LLM again.

"Means the same thing" = the cosine similarity of the two question embeddings
is above a threshold. e.g. "How much is Plus?" and "What does the Plus plan
cost?" should hit the same entry.

A key-term guard then blocks look-alike questions with different answers,
e.g. "trash on the Plus plan" vs "trash on the Free plan".
"""
import re
import time
from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class CacheEntry:
    query: str
    answer: str
    model: str
    created_at: float = field(default_factory=time.time)
    hits: int = 0


class SemanticCache:
    def __init__(self, threshold=config.CACHE_THRESHOLD, ttl=config.CACHE_TTL_SECONDS,
                 max_entries=config.CACHE_MAX_ENTRIES):
        self.threshold = threshold
        self.ttl = ttl
        self.max_entries = max_entries
        self.entries: list[CacheEntry] = []
        self.vectors = np.zeros((0, 0))

    def lookup(self, query, query_vec):
        """Return (entry or None, best similarity seen)."""
        self._evict_expired()
        if not self.entries:
            return None, 0.0
        sims = self.vectors @ query_vec
        order = np.argsort(-sims)
        for i in order:
            if sims[i] < self.threshold:
                break
            if key_terms(self.entries[i].query) == key_terms(query):
                self.entries[i].hits += 1
                return self.entries[i], float(sims[i])
        return None, float(sims[order[0]])

    def add(self, query, query_vec, answer, model):
        if len(self.entries) >= self.max_entries:  # drop the oldest entry
            self.entries.pop(0)
            self.vectors = self.vectors[1:]
        self.entries.append(CacheEntry(query, answer, model))
        row = query_vec[None, :]
        self.vectors = row if self.vectors.size == 0 else np.vstack([self.vectors, row])

    def clear(self):
        self.entries, self.vectors = [], np.zeros((0, 0))

    def _evict_expired(self):
        now = time.time()
        keep = [i for i, e in enumerate(self.entries) if now - e.created_at < self.ttl]
        if len(keep) != len(self.entries):
            self.entries = [self.entries[i] for i in keep]
            self.vectors = self.vectors[keep] if keep else np.zeros((0, 0))


def key_terms(text):
    """Plan names, platforms and numbers mentioned in a question."""
    words = set(re.findall(r"[a-z0-9.]+", text.lower()))
    return {t for t in config.CACHE_KEY_TERMS if t in words} | {w for w in words if w[0].isdigit()}
