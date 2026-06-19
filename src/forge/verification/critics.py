"""Critics that evaluate generated samples for quality issues.

Each critic inspects a :class:`~forge.core.models.Sample` and returns a
:class:`Critique` describing any issues found.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from forge.core.models import Document, Sample
from forge.utils.logging import get_logger
from forge.verification.models import Critique

logger = get_logger(__name__)

# Schema of required content keys per format name.
_FORMAT_KEYS: dict[str, list[str]] = {
    "reasoning": ["question", "analysis", "answer", "metadata"],
    "instruction": ["messages"],
    "agent": ["observation", "thought", "action", "result"],
    "coding": ["issue", "investigation", "patch", "verification"],
    "chat": ["conversations"],
}


class Critic(ABC):
    """Abstract base class for sample critics."""

    @abstractmethod
    async def critique(
        self,
        sample: Sample,
        source_docs: list[Document],
    ) -> Critique:
        """Analyse *sample* against *source_docs* and return a critique."""


class LLMCritic(Critic):
    """Uses an LLM (via *litellm*) to critique a sample.

    Parameters
    ----------
    model:
        The model identifier understood by ``litellm`` (e.g.
        ``"gpt-4o-mini"``).
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    async def critique(
        self,
        sample: Sample,
        source_docs: list[Document],
    ) -> Critique:
        """Prompt the LLM to find issues with the sample."""
        import litellm  # late import to avoid hard dep at module level

        sample_id = sample.lineage.sample_id

        source_text = "\n---\n".join(
            doc.content[:2000] for doc in source_docs
        )
        sample_text = json.dumps(sample.content, indent=2, default=str)

        prompt = (
            "You are a strict quality-assurance reviewer for AI training data.\n"
            "Given the SOURCE DOCUMENTS and a GENERATED SAMPLE, identify any issues.\n\n"
            "SOURCE DOCUMENTS:\n"
            f"{source_text}\n\n"
            "GENERATED SAMPLE:\n"
            f"{sample_text}\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"issues": ["..."], "severity": "none|minor|major|fatal", '
            '"reasoning": "...", "passed": true/false}'
        )

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content or "{}"  # type: ignore[union-attr]
            # Strip markdown fences if present.
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                raw = raw.rsplit("```", 1)[0]
            data = json.loads(raw)
            return Critique(
                sample_id=sample_id,
                critic_model=self.model,
                issues=data.get("issues", []),
                severity=data.get("severity", "none"),
                reasoning=data.get("reasoning", ""),
                passed=data.get("passed", True),
            )
        except Exception as exc:
            logger.warning("LLMCritic failed for %s: %s", sample_id, exc)
            return Critique(
                sample_id=sample_id,
                critic_model=self.model,
                issues=[f"LLM critique error: {exc}"],
                severity="major",
                reasoning=str(exc),
                passed=False,
            )


class FactualCritic(Critic):
    """Checks factual grounding via simple text-overlap heuristics.

    For each sentence-like fragment in the sample answer, this critic
    verifies that a significant portion of its words appear in the source
    documents.  No LLM call is made.
    """

    MIN_OVERLAP: float = 0.3  # minimum word-overlap ratio

    async def critique(
        self,
        sample: Sample,
        source_docs: list[Document],
    ) -> Critique:
        """Return a critique based on word-overlap with source docs."""
        sample_id = sample.lineage.sample_id
        issues: list[str] = []

        # Build a set of normalised source words.
        source_words: set[str] = set()
        for doc in source_docs:
            source_words.update(doc.content.lower().split())

        # Check overlap for answer-like fields.
        answer_text = ""
        for key in ("answer", "result", "analysis"):
            val = sample.content.get(key)
            if isinstance(val, str):
                answer_text += " " + val

        if not answer_text.strip():
            issues.append("No answer/result text found in sample content")
        else:
            answer_words = answer_text.lower().split()
            if answer_words:
                overlap = sum(1 for w in answer_words if w in source_words)
                ratio = overlap / len(answer_words)
                if ratio < self.MIN_OVERLAP:
                    issues.append(
                        f"Low factual grounding: {ratio:.0%} word overlap "
                        f"(threshold {self.MIN_OVERLAP:.0%})"
                    )

        severity = "none" if not issues else "major"
        return Critique(
            sample_id=sample_id,
            critic_model="factual_overlap",
            issues=issues,
            severity=severity,
            reasoning=f"Word-overlap check with {len(source_docs)} source docs",
            passed=len(issues) == 0,
        )


class FormatCritic(Critic):
    """Validates that a sample's ``content`` dict has the required keys.

    The expected keys are determined by the sample's ``lineage.format``
    field.  No LLM call is made.
    """

    async def critique(
        self,
        sample: Sample,
        source_docs: list[Document],
    ) -> Critique:
        """Return a critique based on missing content keys."""
        sample_id = sample.lineage.sample_id
        fmt = sample.lineage.format or "reasoning"
        required = _FORMAT_KEYS.get(fmt, _FORMAT_KEYS["reasoning"])

        missing = [k for k in required if k not in sample.content]
        issues = [f"Missing required key '{k}' for format '{fmt}'" for k in missing]

        severity = "none"
        if missing:
            severity = "fatal" if len(missing) > len(required) // 2 else "major"

        return Critique(
            sample_id=sample_id,
            critic_model="format_check",
            issues=issues,
            severity=severity,
            reasoning=f"Format '{fmt}' requires keys: {required}",
            passed=len(issues) == 0,
        )
