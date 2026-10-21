"""Log every request to a small SQLite database and compute summary stats
(cost, p50/p95 latency, cache hit rate) for the dashboard and benchmark."""
import sqlite3
import time

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    run TEXT NOT NULL,           -- e.g. 'live', 'baseline', 'optimized'
    query TEXT NOT NULL,
    cache_hit INTEGER NOT NULL,  -- 1 if answered from the semantic cache
    similarity REAL,             -- best cache similarity seen
    model TEXT,                  -- model that produced the answer
    tier TEXT,                   -- 'small', 'large', or 'cache'
    route_reason TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL
)"""


class MetricsStore:
    def __init__(self, path=config.DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(SCHEMA)

    def log(self, **row):
        row.setdefault("ts", time.time())
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self.conn.execute(f"INSERT INTO requests ({cols}) VALUES ({marks})", list(row.values()))
        self.conn.commit()

    def clear_run(self, run):
        self.conn.execute("DELETE FROM requests WHERE run = ?", (run,))
        self.conn.commit()

    def runs(self):
        return [r[0] for r in self.conn.execute("SELECT DISTINCT run FROM requests ORDER BY run")]

    def rows(self, run=None, limit=None):
        sql = "SELECT * FROM requests" + (" WHERE run = ?" if run else "") + " ORDER BY id"
        if limit:
            sql = f"SELECT * FROM ({sql.replace('ORDER BY id', 'ORDER BY id DESC')} LIMIT {int(limit)}) ORDER BY id"
        return [dict(r) for r in self.conn.execute(sql, (run,) if run else ())]

    def summary(self, run=None):
        return summarise(self.rows(run))


def summarise(rows):
    if not rows:
        return {"requests": 0}
    lat = np.array([r["latency_ms"] for r in rows])
    cost = sum(r["cost_usd"] for r in rows)
    tiers = {t: sum(1 for r in rows if r["tier"] == t) for t in ("cache", "small", "large")}
    n = len(rows)
    return {
        "requests": n,
        "total_cost_usd": round(cost, 6),
        "cost_per_1k_requests_usd": round(cost / n * 1000, 4),
        "p50_latency_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 1),
        "mean_latency_ms": round(float(lat.mean()), 1),
        "cache_hit_rate": round(tiers["cache"] / n, 4),
        "tier_counts": tiers,
    }
