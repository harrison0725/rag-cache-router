"""Turn text into vectors so we can measure how similar two texts are.

Vectors are L2-normalised, so a plain dot product == cosine similarity.
"""
import hashlib
import re

import numpy as np

from . import config


class FastEmbedder:
    """Real semantic embeddings. Runs on your CPU, costs nothing per call.

    The first run downloads a ~130 MB model from Hugging Face.
    """

    name = "fastembed/bge-small-en-v1.5"

    def __init__(self):
        from fastembed import TextEmbedding

        self.model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def embed_passages(self, texts):
        return _normalise(np.array(list(self.model.passage_embed(texts))))

    def embed_query(self, text):
        return _normalise(np.array(list(self.model.query_embed([text]))))[0]


class HashEmbedder:
    """Offline fallback: hashes words and character trigrams into a vector.

    It catches near-identical wording but NOT true paraphrases, so only use it
    for testing the plumbing without internet access.
    """

    name = "hash (offline test mode)"
    dims = 4096

    def _vec(self, text):
        v = np.zeros(self.dims)
        words = re.findall(r"[a-z0-9]+", text.lower())
        grams = words + [w[i : i + 3] for w in words for i in range(max(1, len(w) - 2))]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.dims] += 1.0
        return v

    def embed_passages(self, texts):
        return _normalise(np.array([self._vec(t) for t in texts]))

    def embed_query(self, text):
        return _normalise(self._vec(text)[None, :])[0]


def _normalise(m):
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


def get_embedder():
    if config.EMBEDDER == "hash":
        return HashEmbedder()
    return FastEmbedder()
