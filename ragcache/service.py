"""The RAG service: question in, answer out, with optional semantic caching
and model routing. Every request is measured and logged."""
import time
from dataclasses import dataclass

from . import config
from .cache import SemanticCache
from .embeddings import get_embedder
from .llm import ClaudeLLM, MockLLM
from .metrics import MetricsStore
from .retriever import Retriever
from .router import route


@dataclass
class Answer:
    question: str
    answer: str
    model: str
    tier: str
    cache_hit: bool
    similarity: float
    route_reason: str
    cost_usd: float
    latency_ms: float
    sources: list[str]
    cached_query: str | None = None  # on a cache hit: the earlier question it matched


class RAGService:
    def __init__(self, use_cache=True, use_routing=True, mock=False, metrics=None, large_model=None):
        self.use_cache = use_cache
        self.use_routing = use_routing
        self.large_model = large_model or config.LARGE_MODEL
        self.embedder = get_embedder()
        self.retriever = Retriever(self.embedder)
        self.cache = SemanticCache()
        self.llm = MockLLM() if mock else ClaudeLLM()
        self.metrics = metrics or MetricsStore()

    def ask(self, question: str, run: str = "live") -> Answer:
        start = time.perf_counter()
        qvec = self.embedder.embed_query(question)

        # 1) Semantic cache: same meaning as an earlier question? Reuse it.
        similarity = 0.0
        if self.use_cache:
            entry, similarity = self.cache.lookup(question, qvec)
            if entry:
                latency = (time.perf_counter() - start) * 1000
                return self._finish(run, Answer(
                    question, entry.answer, entry.model, "cache", True, similarity,
                    f"cache hit (similar to: {entry.query!r})", 0.0, latency, [], entry.query))

        # 2) Retrieve the most relevant help-center chunks.
        chunks = self.retriever.search(qvec)

        # 3) Route: small model for easy questions, large model for hard ones.
        if self.use_routing:
            decision = route(question)
            tier, reason = decision.tier, "; ".join(decision.reasons)
            model = self.large_model if tier == "large" else decision.model
        else:
            model, tier, reason = self.large_model, "large", "routing off (baseline)"

        # 4) Generate the answer.
        pre_llm_ms = (time.perf_counter() - start) * 1000
        result = self.llm.generate(model, question, chunks, tier)
        if self.use_cache:
            self.cache.add(question, qvec, result.text, model)

        # Latency = everything the user waits for (embed + retrieve + LLM).
        latency = pre_llm_ms + result.latency_ms
        return self._finish(run, Answer(
            question, result.text, model, tier, False, similarity, reason,
            result.cost_usd, latency, sorted({c.source for c, _ in chunks}),
        ), result.input_tokens, result.output_tokens)

    def _finish(self, run, a: Answer, input_tokens=0, output_tokens=0):
        self.metrics.log(
            run=run, query=a.question, cache_hit=int(a.cache_hit), similarity=a.similarity,
            model=a.model, tier=a.tier, route_reason=a.route_reason,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=a.cost_usd, latency_ms=a.latency_ms,
        )
        return a

