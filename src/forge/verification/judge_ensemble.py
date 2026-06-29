"""Judge ensemble — 2+ independent models verdict the same sample so that
acceptance requires cross-model agreement instead of a single model's
opinion.

Used in place of (or alongside) the single ``LLMCritic`` for samples that
have already survived programmatic checks and self-consistency selection
in the Generate stage. Final confidence is the *minimum* across judges
(not the average) so one lenient judge can't rescue a sample another judge
flagged as broken, and a fatal verdict from any single judge rejects the
sample outright.
"""

from __future__ import annotations

import asyncio
import json

from forge.core.models import Sample
from forge.providers.llm import LLMProvider
from forge.utils.logging import get_logger
from forge.verification.models import JudgeVerdict

logger = get_logger(__name__)

_JUDGE_PROMPT = (
    "You are an independent quality judge for an AI training sample.\n"
    "Evaluate the reasoning chain step by step. If any step is logically "
    "invalid, factually wrong, or unsupported, the sample fails even if "
    "the final answer happens to be correct.\n\n"
    "SAMPLE:\n{sample_text}\n\n"
    "Respond ONLY with valid JSON:\n"
    '{{"verdict": "accept|reject|revise", "severity": "none|minor|major|fatal", '
    '"step_failures": [<step numbers, if any>], "confidence": 0.0, "reasoning": "..."}}'
)


def merge_judge_verdicts(verdicts: list[JudgeVerdict]) -> tuple[str, float]:
    """Merge independent judge verdicts under a strict agreement rule:
    any fatal severity rejects outright; confidence is the minimum across
    judges (not the average) so one lenient judge can't rescue a sample
    another judge flagged as broken.
    """
    if any(v.severity == "fatal" for v in verdicts):
        return "reject", 0.0

    confidence = min(v.confidence for v in verdicts)
    if all(v.verdict == "accept" for v in verdicts):
        return "accept", confidence

    if any(v.verdict == "reject" for v in verdicts):
        return "reject", confidence

    return "revise", confidence


class JudgeEnsemble:
    """Runs N independent judge models against one sample and merges
    their verdicts under a strict agreement rule.
    """

    def __init__(self, models: list[str], llm: LLMProvider) -> None:
        if len(models) < 2:
            raise ValueError(
                "JudgeEnsemble requires at least 2 models — use a single "
                "LLMCritic instead if you only have one judge model."
            )
        self.models = models
        # All judge calls route through LLMProvider so costs are tracked in
        # CostBudget and visible in cost reports / dry-run estimates.
        self._llm = llm

    async def evaluate(self, sample: Sample) -> tuple[list[JudgeVerdict], str, float]:
        """Return (per-judge verdicts, merged final_verdict, merged confidence)."""
        verdicts = await asyncio.gather(
            *(self._judge(sample, model) for model in self.models)
        )
        verdict, confidence = merge_judge_verdicts(list(verdicts))
        return list(verdicts), verdict, confidence

    async def _judge(self, sample: Sample, model: str) -> JudgeVerdict:
        sample_id = sample.lineage.sample_id
        sample_text = json.dumps(sample.content, indent=2, default=str)
        prompt = _JUDGE_PROMPT.format(sample_text=sample_text)

        try:
            # Route through LLMProvider so token usage and cost are recorded
            # in CostBudget — direct litellm calls bypassed budget tracking.
            raw = await self._llm.complete(
                prompt=prompt,
                model=model,
                temperature=0.0,
                stage="verify",
            )
            text = raw.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(text)
            return JudgeVerdict(
                sample_id=sample_id,
                judge_model=model,
                verdict=data.get("verdict", "revise"),
                severity=data.get("severity", "none"),
                step_failures=data.get("step_failures", []),
                confidence=float(data.get("confidence", 0.0)),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as exc:
            logger.warning("Judge %s failed for %s: %s", model, sample_id, exc)
            # A failed judge call counts as a non-vote at zero confidence,
            # not a free pass — it pulls the min-confidence merge down
            # rather than being silently dropped.
            return JudgeVerdict(
                sample_id=sample_id,
                judge_model=model,
                verdict="revise",
                severity="none",
                confidence=0.0,
                reasoning=f"Judge call failed: {exc}",
            )
