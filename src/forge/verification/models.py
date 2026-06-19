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
