"""Pipeline context — shared state passed between stages."""

from __future__ import annotations

from uuid import uuid4

from forge.core.budget import CostBudget
from forge.core.config import ForgeSettings
from forge.core.models import (
    AnalysisResult,
    CleanedDocument,
    DatasetMetrics,
    DeduplicationReport,
    DiversityScore,
    Document,
    GenerationPlan,
    RunMetadata,
    RunStatus,
    Sample,
)
from forge.metrics.collector import MetricsCollector
from forge.storage.base import StorageBackend


class PipelineContext:
    """Mutable state container threaded through every stage."""

    def __init__(
        self,
        topic: str,
        settings: ForgeSettings,
        storage: StorageBackend,
        metrics: MetricsCollector,
        budget: CostBudget,
    ) -> None:
        self.run_id: str = str(uuid4())
        self.topic = topic
        self.settings = settings
        self.storage = storage
        self.metrics = metrics
        self.budget = budget

        # Stage outputs
        self.documents: list[Document] = []
        self.cleaned_documents: list[CleanedDocument] = []
        self.analysis: AnalysisResult | None = None
        self.generation_plan: GenerationPlan | None = None
        self.samples: list[Sample] = []

        # Quality reports
        self.dedup_report: DeduplicationReport | None = None
        self.diversity_score: DiversityScore | None = None

        # Run tracking
        self.run_metadata = RunMetadata(run_id=self.run_id, topic=topic, status=RunStatus.PENDING)
        self.dry_run: bool = False
        self.template_name: str = ""

    def __repr__(self) -> str:
        return (
            f"<PipelineContext topic={self.topic!r} docs={len(self.documents)} "
            f"samples={len(self.samples)} run={self.run_id[:8]}>"
        )
