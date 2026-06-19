"""Verification package — critics, scorers, validators, and consensus.

This package evaluates generated training samples for quality, factual
grounding, and format compliance before they are included in a dataset.
"""

from __future__ import annotations

from forge.verification.consensus import ConsensusEngine
from forge.verification.critics import Critic, FactualCritic, FormatCritic, LLMCritic
from forge.verification.models import ConsensusResult, Critique, ScoreResult
from forge.verification.scorers import LLMScorer, Scorer
from forge.verification.validators import SampleValidator

__all__ = [
    # Critics
    "Critic",
    "LLMCritic",
    "FactualCritic",
    "FormatCritic",
    # Validators
    "SampleValidator",
    # Scorers
    "Scorer",
    "LLMScorer",
    # Consensus
    "ConsensusEngine",
    # Models
    "Critique",
    "ScoreResult",
    "ConsensusResult",
]
