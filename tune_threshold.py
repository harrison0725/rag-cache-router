"""Pick a safe semantic-cache threshold, for free (no LLM calls).

For every pair of questions in eval/questions.json it computes the embedding
similarity and checks whether the pair is the SAME question (a paraphrase, so a
cache hit is good) or a DIFFERENT question (a cache hit would return the wrong
answer). It then shows, for each threshold, how many good hits you'd get and
how many wrong ones.

    python tune_threshold.py
"""
import itertools
import json
from pathlib import Path

import numpy as np

from ragcache import config
from ragcache.cache import key_terms
from ragcache.embeddings import get_embedder


def main():
    questions = json.loads((Path(__file__).parent / "eval" / "questions.json").read_text())
    texts = [(q["id"], t) for q in questions for t in [q["question"], *q["paraphrases"]]]
    emb = get_embedder()
    print(f"Embedder: {emb.name}\n")
    vecs = np.array([emb.embed_query(t) for _, t in texts])

    same, diff = [], []
    for i, j in itertools.combinations(range(len(texts)), 2):
        sim = float(vecs[i] @ vecs[j])
        (same if texts[i][0] == texts[j][0] else diff).append((sim, texts[i][1], texts[j][1]))

    guarded = lambda a, b: key_terms(a) == key_terms(b)
    print(f"{'threshold':>9} | {'paraphrases caught':>18} | {'wrong matches':>13} | {'wrong after key-term guard':>26}")
    for t in np.arange(0.80, 0.991, 0.01):
        good = sum(s >= t and guarded(a, b) for s, a, b in same)
        bad = sum(s >= t for s, *_ in diff)
        bad_guarded = sum(s >= t and guarded(a, b) for s, a, b in diff)
        mark = "  <- current" if abs(t - config.CACHE_THRESHOLD) < 1e-6 else ""
        print(f"{t:>9.2f} | {good:>8} / {len(same):<8} | {bad:>13} | {bad_guarded:>26}{mark}")

    print("\nMost similar pairs of DIFFERENT questions (the risky ones):")
    for s, a, b in sorted(diff, reverse=True)[:5]:
        print(f"  {s:.3f}  {a!r}  vs  {b!r}")
    print("\nLeast similar paraphrase pairs (the ones you'd miss):")
    for s, a, b in sorted(same)[:5]:
        print(f"  {s:.3f}  {a!r}  vs  {b!r}")
    print("\nPick the lowest threshold with 0 wrong matches after the guard, then set CACHE_THRESHOLD in .env.")


if __name__ == "__main__":
    main()
