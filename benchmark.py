"""Before/after benchmark: baseline (every question -> large model, no cache)
vs optimized (semantic cache + model routing), on the same realistic workload.

    python benchmark.py            # real run with Claude (costs roughly $1-3)
    python benchmark.py --mock     # free dry run with a fake LLM, numbers NOT real

Writes results/benchmark.md (quote-ready table), results/benchmark.json and
results/requests.csv. Both runs also show up in the dashboard.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

from ragcache import config
from ragcache.metrics import MetricsStore, summarise
from ragcache.service import RAGService

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def build_workload(questions, seed=7, repeat_rate=0.5):
    """Real support traffic repeats itself: the same question, reworded, or
    asked again word for word. The workload mixes: every original question once,
    every paraphrase once, and ~repeat_rate of questions asked again verbatim."""
    rng = random.Random(seed)
    items = []
    for q in questions:
        items.append((q["id"], q["question"]))
        items += [(q["id"], p) for p in q["paraphrases"]]
        if rng.random() < repeat_rate:
            items.append((q["id"], q["question"]))
    rng.shuffle(items)
    return items


def fact_recall(answer, facts):
    """Share of required facts that appear in the answer (each fact may list
    alternatives separated by '|'). A cheap, transparent quality check."""
    a = answer.lower()
    hits = sum(any(alt.strip() in a for alt in f.lower().split("|")) for f in facts)
    return hits / len(facts)


def run_mode(name, workload, by_id, query_to_id, use_cache, use_routing, mock, store):
    store.clear_run(name)
    svc = RAGService(use_cache=use_cache, use_routing=use_routing, mock=mock, metrics=store)
    records = []
    for i, (qid, text) in enumerate(workload, 1):
        a = svc.ask(text, run=name)
        q = by_id[qid]
        wrong_hit = a.cache_hit and query_to_id.get(a.cached_query) != qid
        records.append({
            "run": name, "id": qid, "difficulty": q["difficulty"], "question": text,
            "tier": a.tier, "model": a.model, "cache_hit": a.cache_hit,
            "similarity": round(a.similarity, 4), "matched": a.cached_query,
            "wrong_cache_hit": wrong_hit, "cost_usd": a.cost_usd,
            "latency_ms": round(a.latency_ms, 1),
            "quality": round(fact_recall(a.answer, q["facts"]), 3),
            "route_reason": a.route_reason, "answer": a.answer,
        })
        flag = "CACHE" if a.cache_hit else a.tier.upper()
        print(f"  [{name} {i:>3}/{len(workload)}] {flag:<5} {a.latency_ms:>7.0f} ms  ${a.cost_usd:.5f}  {text[:60]}")
    return records


def report(base, opt, mock):
    def stats(recs):
        s = summarise(recs)
        s["avg_quality"] = round(sum(r["quality"] for r in recs) / len(recs), 3)
        s["wrong_cache_hits"] = sum(r["wrong_cache_hit"] for r in recs)
        routed = [r for r in recs if r["tier"] in ("small", "large")]
        if routed and any(r["tier"] == "small" for r in recs):
            ok = sum((r["tier"] == "large") == (r["difficulty"] == "hard") for r in routed)
            s["router_accuracy"] = round(ok / len(routed), 3)
        return s

    b, o = stats(base), stats(opt)
    pct = lambda new, old: (1 - new / old) * 100 if old else 0.0
    cost_cut = pct(o["total_cost_usd"], b["total_cost_usd"])
    p95_cut = pct(o["p95_latency_ms"], b["p95_latency_ms"])

    rows = [
        ("Requests", b["requests"], o["requests"]),
        ("Total LLM cost", f"${b['total_cost_usd']:.4f}", f"${o['total_cost_usd']:.4f}"),
        ("Cost per 1k requests", f"${b['cost_per_1k_requests_usd']:.2f}", f"${o['cost_per_1k_requests_usd']:.2f}"),
        ("p50 latency", f"{b['p50_latency_ms']:.0f} ms", f"{o['p50_latency_ms']:.0f} ms"),
        ("p95 latency", f"{b['p95_latency_ms']:.0f} ms", f"{o['p95_latency_ms']:.0f} ms"),
        ("Cache hit rate", f"{b['cache_hit_rate']:.0%}", f"{o['cache_hit_rate']:.0%}"),
        ("Wrong cache hits", "-", o["wrong_cache_hits"]),
        ("Requests to small / large model", f"{b['tier_counts']['small']} / {b['tier_counts']['large']}",
         f"{o['tier_counts']['small']} / {o['tier_counts']['large']}"),
        ("Router accuracy (vs. labels)", "-", f"{o.get('router_accuracy', 0):.0%}"),
        ("Answer quality (fact recall)", f"{b['avg_quality']:.1%}", f"{o['avg_quality']:.1%}"),
    ]
    lines = []
    if mock:
        lines += ["> **MOCK RUN: these numbers are simulated. Do not quote them.** Run without `--mock` for real numbers.", ""]
    lines += [
        f"# Benchmark: baseline vs. semantic cache + routing",
        "",
        f"Run at {time.strftime('%Y-%m-%d %H:%M')} · small model `{config.SMALL_MODEL}` · large model `{config.LARGE_MODEL}` · "
        f"cache threshold {config.CACHE_THRESHOLD} · embedder: local (free)",
        "",
        "| Metric | Baseline (large model, no cache) | Optimized (cache + routing) |",
        "|---|---|---|",
        *[f"| {m} | {x} | {y} |" for m, x, y in rows],
        "",
        f"**LLM cost: -{cost_cut:.0f}%   ·   p95 latency: -{p95_cut:.0f}%   ·   quality: "
        f"{b['avg_quality']:.1%} → {o['avg_quality']:.1%}**",
        "",
    ]
    if o["avg_quality"] >= b["avg_quality"] - 0.02:
        lines += [
            "Suggested resume line:",
            "",
            f"> Added semantic caching and model routing to a RAG service, reducing LLM cost by {cost_cut:.0f}% "
            f"and p95 latency by {p95_cut:.0f}% while maintaining answer quality "
            f"({o['avg_quality']:.0%} vs. {b['avg_quality']:.0%} fact recall on a {b['requests']}-query benchmark).",
        ]
    else:
        lines += [
            "**Quality dropped by more than 2 points**, so don't claim \"maintaining quality\" yet. "
            "Check `results/requests.csv` for wrong cache hits (raise CACHE_THRESHOLD) or hard "
            "questions sent to the small model (adjust the router), then re-run.",
        ]
    return "\n".join(lines), {"baseline": b, "optimized": o, "cost_reduction_pct": round(cost_cut, 1),
                              "p95_reduction_pct": round(p95_cut, 1), "mock": mock}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="fake LLM, no API key, simulated numbers")
    ap.add_argument("--limit", type=int, help="only run the first N workload items (quick test)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    questions = json.loads((ROOT / "eval" / "questions.json").read_text())
    by_id = {q["id"]: q for q in questions}
    query_to_id = {t: q["id"] for q in questions for t in [q["question"], *q["paraphrases"]]}
    workload = build_workload(questions, seed=args.seed)[: args.limit]
    print(f"Workload: {len(workload)} requests from {len(questions)} distinct questions\n")

    store = MetricsStore()
    print("Baseline: every request -> large model, no cache")
    base = run_mode("baseline", workload, by_id, query_to_id, False, False, args.mock, store)
    print("\nOptimized: semantic cache + model routing")
    opt = run_mode("optimized", workload, by_id, query_to_id, True, True, args.mock, store)

    md, summary = report(base, opt, args.mock)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmark.md").write_text(md + "\n")
    (RESULTS / "benchmark.json").write_text(json.dumps(summary, indent=2))
    with open(RESULTS / "requests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(base[0]))
        w.writeheader()
        w.writerows(base + opt)
    print("\n" + md)
    print(f"\nSaved results/benchmark.md, results/benchmark.json, results/requests.csv")


if __name__ == "__main__":
    main()
