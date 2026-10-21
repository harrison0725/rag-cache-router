"""Cost-aware routing: send easy questions to the small model and hard ones to
the large model.

It's a transparent rule-based scorer, so you can explain every decision:
each signal of "this needs reasoning" adds points, and a score at or above
ROUTER_HARD_THRESHOLD goes to the large model.
"""
import re
from dataclasses import dataclass

from . import config

# Words that usually mean the user wants reasoning, not a single fact.
REASONING_CUES = [
    "why", "compare", "comparison", "difference", "versus", " vs", "trade-off",
    "tradeoff", "pros and cons", "should we", "should i choose", "should i get",
    "should i buy", "recommend", "my options", "our options", "what options", "worth it", "explain", "what happens if", "what happens to",
    "what would happen", "scenario", "strategy", "calculate", "how much would",
    "total cost", "cheaper",
]


@dataclass
class RouteDecision:
    model: str
    tier: str          # "small" or "large"
    score: int
    reasons: list[str]


def route(query: str) -> RouteDecision:
    q = query.lower()
    score, reasons = 0, []

    cues = [c.strip() for c in REASONING_CUES if c in q]
    if cues:
        score += 2
        reasons.append(f"reasoning words: {', '.join(cues[:3])}")

    words = len(re.findall(r"\w+", q))
    if words > 25:
        score += 1
        reasons.append(f"long question ({words} words)")

    question_words = len(re.findall(r"\b(what|how|which|when|why|where)\b", q))
    if question_words >= 3:
        score += 2
        reasons.append(f"multi-part question ({question_words} parts)")
    elif q.count("?") > 1 or question_words == 2:
        score += 1
        reasons.append("two questions in one")

    if re.search(r"\d", q) and re.search(r"\b(users?|people|days?|months?|years?|tb|gb)\b", q):
        score += 1
        reasons.append("numbers to work with")

    if score >= config.ROUTER_HARD_THRESHOLD:
        return RouteDecision(config.LARGE_MODEL, "large", score, reasons)
    return RouteDecision(config.SMALL_MODEL, "small", score, reasons or ["simple lookup"])
