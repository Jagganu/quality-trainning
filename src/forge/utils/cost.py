"""Cost estimation helpers using LiteLLM's cost data."""

from __future__ import annotations

import litellm


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate cost in USD using LiteLLM's pricing data.

    Falls back to a conservative default if model pricing is unknown.
    """
    try:
        cost = litellm.completion_cost(
            model=model,
            prompt="",
            completion="",
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        return cost
    except Exception:
        # Fallback: $0.01 per 1K input, $0.03 per 1K output (conservative)
        return (input_tokens / 1000) * 0.01 + (output_tokens / 1000) * 0.03


def format_cost(cost: float) -> str:
    """Format cost as a human-readable USD string."""
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"
