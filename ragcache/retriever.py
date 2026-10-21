"""Load the help-center docs, split them into chunks, and find the chunks
most relevant to a question (the "R" in RAG)."""
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class Chunk:
    source: str
    text: str


def load_chunks(docs_dir=config.DOCS_DIR, max_chars=config.CHUNK_MAX_CHARS):
    """Split each markdown file into paragraph-based chunks of ~max_chars."""
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        title = path.read_text().splitlines()[0].lstrip("# ").strip()
        paragraphs = [p.strip() for p in path.read_text().split("\n\n") if p.strip()]
        current = ""
        for p in paragraphs:
            if p.startswith("#"):
                continue
            if current and len(current) + len(p) > max_chars:
                chunks.append(Chunk(path.name, f"[{title}]\n{current}"))
                current = ""
            current = f"{current}\n\n{p}".strip()
        if current:
            chunks.append(Chunk(path.name, f"[{title}]\n{current}"))
    return chunks


class Retriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks = load_chunks()
        self.matrix = embedder.embed_passages([c.text for c in self.chunks])

    def search(self, query_vec, k=config.TOP_K):
        scores = self.matrix @ query_vec
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
