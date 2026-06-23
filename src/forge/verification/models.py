"""Verification-specific data models.

These models capture the output of critics, scorers, and the consensus
engine.  They are distinct from the core ``VerificationResult`` model
which summarises the final outcome attached to a ``Sample``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Critique(BaseModel):
    """Structured critique produced by a :class:`Critic`."""

    sample_id: str
    critic_model: str
    issues: list[str] = Field(default_factory=list)
    severity: str = "none"  # none | minor | major | fatal
    reasoning: str = ""
    passed: bool = True


class ScoreResult(BaseModel):
    """Numeric quality scores assigned by a :class:`Scorer`."""

    sample_id: str
    scorer_model: str = ""
    accuracy: float = 0.0
    relevance: float = 0.0
    completeness: float = 0.0
    overall: float = 0.0


class ConsensusResult(BaseModel):
    """Aggregated verdict from multiple critiques and scores."""

    sample_id: str
    critiques: list[Critique] = Field(default_factory=list)
    scores: list[ScoreResult] = Field(default_factory=list)
    final_verdict: str = "accept"  # accept | reject | revise
    confidence: float = 0.0
    reasoning: str = ""
    # Populated only by ConsensusEngine.evaluate_with_judges — empty/0.0
    # for the original single-critic evaluate() path.
    judge_verdicts: list[JudgeVerdict] = Field(default_factory=list)
    agreement_ratio: float = 0.0


class JudgeVerdict(BaseModel):
    """One independent judge model's opinion on a single (already-selected)
    candidate sample.

    Used by :class:`~forge.verification.judge_ensemble.JudgeEnsemble` to
    collect opinions from 2+ different model families before merging them
    into a single :class:`ConsensusResult`. Distinct from :class:`Critique`
    in that a verdict always carries an explicit accept/reject/revise call
    plus which reasoning steps (if any) the judge flagged as broken,
    rather than a free-form issue list.
    """

    sample_id: str
    judge_model: str
    verdict: str = "accept"  # accept | reject | revise
    severity: str = "none"  # none | minor | major | fatal
    step_failures: list[int] = Field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""


ConsensusResult.model_rebuild()
