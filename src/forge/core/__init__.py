"""All Pydantic data models for ForgeGravity.

This module defines every structured data type used across the pipeline,
including data lineage, quality metrics, and the v3 reasoning format.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime.

    Replaces ``datetime.utcnow()`` which is deprecated in Python 3.12+
    and returns a naive datetime, leading to subtle comparison bugs.
    """
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    GATE_FAILED = "gate_failed"


class Difficulty(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class SampleVerdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REVISE = "revise"


# ---------------------------------------------------------------------------
# Documents (Collect stage)
# ---------------------------------------------------------------------------

class Document(BaseModel):
    """A raw collected document."""
    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    url: str | None = None
    title: str = ""
    content: str = ""
    content_type: str = "text/html"
    metadata: dict[str, Any] = Field(default_factory=dict)
    collected_at: datetime = Field(default_factory=_utcnow)
    content_hash: str = ""
    word_count: int = 0


class CleanedDocument(BaseModel):
    """A cleaned and chunked document."""
    doc_id: str
    source_doc_id: str
    chunks: list[str] = Field(default_factory=list)
    language: str = "en"
    quality_score: float = 0.0
    word_count: int = 0


# ---------------------------------------------------------------------------
# Analysis (Analyze stage)
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    """Output of the Analyze stage — guides generation."""
    topic: str
    subtopics: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    sample_types: list[str] = Field(default_factory=list)
    generation_plan: list[SubtopicPlan] = Field(default_factory=list)


class SubtopicPlan(BaseModel):
    """Plan for generating samples within a subtopic."""
    name: str
    concepts: list[str] = Field(default_factory=list)
    sample_count: int = 10
    difficulties: dict[str, int] = Field(
        default_factory=lambda: {"beginner": 3, "intermediate": 4, "advanced": 3}
    )


# ---------------------------------------------------------------------------
# Data Lineage
# ---------------------------------------------------------------------------

class SampleLineage(BaseModel):
    """Full traceability for a generated sample."""
    sample_id: str = Field(default_factory=lambda: str(uuid4()))
    source_documents: list[str] = Field(default_factory=list)
    generation_model: str = ""
    generation_timestamp: datetime = Field(default_factory=_utcnow)
    template: str = ""
    pipeline_run_id: str = ""
    format: str = ""
    stage_versions: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Samples (Generate stage) — v3 reasoning format
# ---------------------------------------------------------------------------

class Sample(BaseModel):
    """A generated training sample with full lineage."""
    lineage: SampleLineage = Field(default_factory=SampleLineage)
    content: dict[str, Any] = Field(default_factory=dict)
    verification: VerificationResult | None = None
    quality_score: float | None = None


class RawGeneration(BaseModel):
    """Raw LLM output before formatting."""
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class VerificationResult(BaseModel):
    """Outcome of verifying a single sample."""
    sample_id: str
    verdict: SampleVerdict = SampleVerdict.ACCEPT
    score: float = 0.0
    issues: list[str] = Field(default_factory=list)
    critic_model: str = ""
    scorer_model: str = ""
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class DuplicateResult(BaseModel):
    """Result of checking one document for duplicates."""
    is_duplicate: bool = False
    strategy: str = ""       # "exact" | "hash" | "simhash"
    duplicate_of: str | None = None
    similarity: float = 0.0


class DeduplicationReport(BaseModel):
    """Summary of deduplication across a corpus."""
    total_processed: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    unique_documents: int = 0
    duplicate_pairs: list[tuple[str, str, float]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

class DiversityScore(BaseModel):
    """Measures how diverse a dataset is."""
    topic_entropy: float = 0.0
    difficulty_balance: float = 0.0
    prompt_uniqueness: float = 0.0
    overall: float = 0.0


# ---------------------------------------------------------------------------
# Generation Plan
# ---------------------------------------------------------------------------

class GenerationPlan(BaseModel):
    """Full plan for generating a dataset."""
    topic: str
    subtopics: list[SubtopicPlan] = Field(default_factory=list)
    total_samples: int = 0
    format: str = "reasoning"
    estimated_cost: float = 0.0


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

class CostEstimate(BaseModel):
    """Estimated cost for a pipeline operation."""
    estimated_tokens: int = 0
    estimated_cost: float = 0.0
    model: str = ""
    confidence: str = "low"   # "low" | "medium" | "high"


class CostReport(BaseModel):
    """Actual cost incurred during a pipeline run."""
    total_cost: float = 0.0
    cost_by_stage: dict[str, float] = Field(default_factory=dict)
    cost_by_model: dict[str, float] = Field(default_factory=dict)
    total_tokens_in: int = 0
    total_tokens_out: int = 0


# ---------------------------------------------------------------------------
# Run Metadata
# ---------------------------------------------------------------------------

class RunMetadata(BaseModel):
    """Metadata for a pipeline run."""
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str = ""
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    status: RunStatus = RunStatus.PENDING
    stages_completed: list[str] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class VerificationReport(BaseModel):
    """Aggregate verification results."""
    total_verified: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    failure_reasons: dict[str, int] = Field(default_factory=dict)


class DatasetMetrics(BaseModel):
    """Complete quality metrics for a dataset."""
    total_samples: int = 0
    verified_samples: int = 0
    rejected_samples: int = 0
    diversity_score: DiversityScore = Field(default_factory=DiversityScore)
    deduplication_report: DeduplicationReport = Field(default_factory=DeduplicationReport)
    cost_report: CostReport = Field(default_factory=CostReport)
    verification_report: VerificationReport = Field(default_factory=VerificationReport)
    stage_durations: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dry Run
# ---------------------------------------------------------------------------

class DryRunPlan(BaseModel):
    """Output of --dry-run: estimated scope and cost without executing."""
    topic: str
    template: str = ""
    format: str = "reasoning"
    estimated_pages: int = 0
    estimated_documents: int = 0
    estimated_samples: int = 0
    estimated_tokens: int = 0
    estimated_cost: float = 0.0
    estimated_runtime: str = ""
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Quality Gates
# ---------------------------------------------------------------------------

class GateResult(BaseModel):
    """Result of checking a single quality gate."""
    gate: str
    passed: bool
    actual_value: float
    threshold: float
    message: str = ""
