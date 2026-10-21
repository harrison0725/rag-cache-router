"""Call Claude and measure tokens, cost, and latency for every call."""
import time
from dataclasses import dataclass

from . import config

SYSTEM_PROMPT = """You are the support assistant for Lumen Drive, a cloud storage service.
Answer the customer's question using only the help-center excerpts provided.
If the excerpts don't contain the answer, say you don't know and suggest contacting support.
Be concise: a short paragraph or a few bullet points. Include exact numbers, prices, and menu paths when relevant."""


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


def build_prompt(question, chunks):
    context = "\n\n---\n\n".join(c.text for c, _ in chunks)
    return f"<help_center_excerpts>\n{context}\n</help_center_excerpts>\n\nCustomer question: {question}"


class ClaudeLLM:
    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def generate(self, model, question, chunks) -> LLMResult:
        start = time.perf_counter()
        response = self.client.messages.create(
            model=model,
            max_tokens=config.MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            output_config={"effort": config.EFFORT.get(model, "medium")},
            messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if response.stop_reason == "refusal":
            text = "Sorry, I can't help with that request."
        else:
            text = "".join(b.text for b in response.content if b.type == "text").strip()

        # usage.output_tokens includes any thinking tokens, which are billed.
        u = response.usage
        return LLMResult(
            text=text,
            model=model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cost_usd=config.price_of(model, u.input_tokens, u.output_tokens),
            latency_ms=latency_ms,
        )


class MockLLM:
    """Fake LLM for testing the pipeline with no API key. Numbers are SIMULATED
    (large model = slower and pricier) and must not be quoted anywhere."""

    def __init__(self, speedup=20.0):
        self.speedup = speedup

    def generate(self, model, question, chunks) -> LLMResult:
        is_large = model == config.LARGE_MODEL
        time.sleep((2.5 if is_large else 0.7) / self.speedup)
        prompt = build_prompt(question, chunks)
        in_tok = len(prompt) // 4 + 100
        out_tok = 400 if is_large else 150
        text = f"[mock answer from {model}] " + chunks[0][0].text[:300]
        return LLMResult(text, model, in_tok, out_tok, config.price_of(model, in_tok, out_tok),
                         (2500 if is_large else 700) * (0.8 + 0.4 * (hash(question) % 100) / 100))
