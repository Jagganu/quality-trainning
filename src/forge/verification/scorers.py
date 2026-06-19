"""Scorers that assign numeric quality scores to generated samples.

Each scorer evaluates a :class:`~forge.core.models.Sample` and produces a
:class:`ScoreResult` with per-dimension and overall quality metrics.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from forge.core.models import Sample
from forge.utils.logging import get_logger
from forge.verification.models import ScoreResult

logger = get_logger(__name__)

# Default dimension weights for the overall score.
_WEIGHTS = {"accuracy": 0.4, "relevance": 0.3, "completeness": 0.3}


class Scorer(ABC):
    """Abstract base class for sample scorers."""

    @abstractmethod
    async def score(self, sample: Sample) -> ScoreResult:
        """Score *sample* and return a :class:`ScoreResult`."""


class LLMScorer(Scorer):
    """Uses an LLM to assign quality scores to a sample.

    The model is prompted to return JSON with ``accuracy``, ``relevance``,
    and ``completeness`` values in ``[0, 1]``.  The ``overall`` score is
    the weighted average of these three dimensions.

    Parameters
    ----------
    model:
        LLM identifier understood by ``litellm`` (e.g. ``"gpt-4o-mini"``).
    weights:
        Optional dict mapping dimension names to weights.  Defaults to
        ``{"accuracy": 0.4, "relevance": 0.3, "completeness": 0.3}``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        weights: dict[str, float] | None = None,
    ) -> None:
        self.model = model
        self.weights = weights or dict(_WEIGHTS)

    async def score(self, sample: Sample) -> ScoreResult:
        """Prompt the LLM to score the sample."""
        import litellm  # late import to keep dependency optional

        sample_id = sample.lineage.sample_id
        sample_text = json.dumps(sample.content, indent=2, default=str)

        prompt = (
            "You are a data quality evaluator for AI training samples.\n"
            "Score the following sample on three dimensions, each from 0.0 to 1.0:\n"
            "  - accuracy: factual correctness\n"
            "  - relevance: how well the answer addresses the question\n"
            "  - completeness: thoroughness of the answer\n\n"
            "SAMPLE:\n"
            f"{sample_text}\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"accuracy": 0.0, "relevance": 0.0, "completeness": 0.0}'
        )

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content or "{}"  # type: ignore[union-attr]
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                raw = raw.rsplit("```", 1)[0]
            data = json.loads(raw)

            accuracy = float(data.get("accuracy", 0.0))
            relevance = float(data.get("relevance", 0.0))
            completeness = float(data.get("completeness", 0.0))

            overall = (
                self.weights["accuracy"] * accuracy
                + self.weights["relevance"] * relevance
                + self.weights["completeness"] * completeness
            )

            return ScoreResult(
                sample_id=sample_id,
                scorer_model=self.model,
                accuracy=accuracy,
                relevance=relevance,
                completeness=completeness,
                overall=round(overall, 4),
            )
        except Exception as exc:
            logger.warning("LLMScorer failed for %s: %s", sample_id, exc)
            return ScoreResult(
                sample_id=sample_id,
                scorer_model=self.model,
            )
