# Benchmark: baseline vs. semantic cache + routing

Run at 2026-10-21 20:00 · baseline model `claude-opus-5-5` · optimized: small `claude-haiku-5-5`, large `claude-sonnet-5-5` · cache threshold 0.9 · embedder: local (free)

| Metric | Baseline (large model, no cache) | Optimized (cache + routing) |
|---|---|---|
| Requests | 71 | 71 |
| Total LLM cost | $0.7079 | $0.0694 |
| Cost per 1k requests | $9.97 | $0.98 |
| p50 latency | 3130 ms | 914 ms |
| p95 latency | 7001 ms | 3732 ms |
| Cache hit rate | 0% | 41% |
| Wrong cache hits | - | 0 |
| Requests to small / large model | 0 / 71 | 30 / 12 |
| Router accuracy (vs. labels) | - | 100% |
| Answer quality (fact recall) | 92.0% | 91.6% |

**LLM cost: -90%   ·   p95 latency: -47%   ·   quality: 92.0% → 91.6%**

Suggested resume line:

> Added semantic caching and model routing to a RAG service, reducing LLM cost by 90% and p95 latency by 47% while maintaining answer quality (92% vs. 92% fact recall on a 71-query benchmark).
