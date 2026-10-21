"""Web server: an /api/ask endpoint plus a live dashboard at http://localhost:8000

    uvicorn app:app --reload
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ragcache.metrics import MetricsStore
from ragcache.service import RAGService

app = FastAPI(title="RAG with semantic cache + model routing")
metrics = MetricsStore()
service = RAGService(
    use_cache=os.getenv("USE_CACHE", "1") == "1",
    use_routing=os.getenv("USE_ROUTING", "1") == "1",
    mock=os.getenv("MOCK_LLM", "0") == "1",
    metrics=metrics,
)


class AskRequest(BaseModel):
    question: str


@app.post("/api/ask")
def ask(req: AskRequest):
    a = service.ask(req.question, run="live")
    return {
        "answer": a.answer, "model": a.model, "tier": a.tier, "cache_hit": a.cache_hit,
        "similarity": round(a.similarity, 3), "route_reason": a.route_reason,
        "cost_usd": a.cost_usd, "latency_ms": round(a.latency_ms, 1), "sources": a.sources,
    }


@app.get("/api/runs")
def runs():
    return metrics.runs()


@app.get("/api/summary")
def summary(run: str | None = None):
    return metrics.summary(run)


@app.get("/api/requests")
def requests(run: str | None = None, limit: int = 500):
    return metrics.rows(run, limit)


@app.get("/")
def dashboard():
    return FileResponse(Path(__file__).parent / "static" / "dashboard.html")
