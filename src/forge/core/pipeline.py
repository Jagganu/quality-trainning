"""Pipeline orchestrator — runs stages in order with hooks, budget, and gates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from forge.core.budget import BudgetExceededError, CostBudget
from forge.core.config import ForgeSettings, load_settings
from forge.core.context import PipelineContext
from forge.core.gates import QualityGate, QualityGateFailedError
from forge.core.hooks import HookEvent, HookManager
from forge.core.models import (
    DatasetMetrics,
    DryRunPlan,
    RunMetadata,
    RunStatus,
    Sample,
)
from forge.core.stage import Stage
from forge.metrics.collector import MetricsCollector
from forge.providers.llm import LLMProvider
from forge.storage.filesystem import FilesystemStorage
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class PipelineResult:
    """What :meth:`Pipeline.run` returns."""

    def __init__(
        self,
        run_metadata: RunMetadata,
        metrics: DatasetMetrics,
        samples: list[Sample],
        output_dir: str,
    ) -> None:
        self.run_metadata = run_metadata
        self.metrics = metrics
        self.samples = samples
        self.output_dir = output_dir


class Pipeline:
    """The main orchestrator — thread stages together with hooks and guards."""

    def __init__(self, settings: ForgeSettings) -> None:
        self._settings = settings
        self._stages: list[Stage] = []
        self._hooks = HookManager()
        self._storage = FilesystemStorage(Path(settings.output_dir) / ".forge_data")
        self._metrics = MetricsCollector()
        self._budget = CostBudget(max_cost=settings.budget.max_cost_usd)
        self._gates = QualityGate(settings.quality_gates)
        self._llm = LLMProvider(settings, self._budget)
        self._setup_default_stages()

    # ------------------------------------------------------------------ #
    # Stage management
    # ------------------------------------------------------------------ #

    def add_stage(self, stage: Stage) -> None:
        """Append a stage to the pipeline."""
        self._stages.append(stage)
        logger.debug("Stage added: %s", stage.name)

    # ------------------------------------------------------------------ #
    # Full run
    # ------------------------------------------------------------------ #

    async def run(
        self,
        topic: str,
        template_name: str = "",
        dry_run: bool = False,
    ) -> PipelineResult:
        """Execute every stage in order."""
        if dry_run:
            plan = await self.dry_run(topic)
            # Return a lightweight result so CLI can display it
            return PipelineResult(
                run_metadata=RunMetadata(topic=topic, status=RunStatus.COMPLETED),
                metrics=self._metrics.snapshot(),
                samples=[],
                output_dir="",
            )

        context = PipelineContext(
            topic=topic,
            settings=self._settings,
            storage=self._storage,
            metrics=self._metrics,
            budget=self._budget,
        )
        context.template_name = template_name or topic
        context.run_metadata.status = RunStatus.RUNNING

        await self._hooks.emit(HookEvent.PIPELINE_START, topic=topic, context=context)

        try:
            for stage in self._stages:
                # Pre-flight
                if not await stage.validate(context):
                    logger.info("Stage %s skipped (validation returned False)", stage.name)
                    continue

                logger.info("▶ Starting stage: %s", stage.name)
                timer = self._metrics.timer(stage.name)
                await self._hooks.emit(HookEvent.BEFORE_STAGE, stage=stage.name, context=context)

                with timer:
                    context = await stage.run(context)

                context.run_metadata.stages_completed.append(stage.name)
                await self._hooks.emit(
                    HookEvent.AFTER_STAGE, stage=stage.name,
                    context=context, duration=timer.elapsed,
                )
                logger.info("✓ Stage %s completed (%.1fs)", stage.name, timer.elapsed)

                # Checkpoint
                await self._checkpoint(stage.name, context)

        except BudgetExceededError as exc:
            context.run_metadata.status = RunStatus.BUDGET_EXCEEDED
            await self._hooks.emit(HookEvent.ON_ERROR, error=exc, context=context)
            logger.error("Budget exceeded: %s", exc)
            raise
        except Exception as exc:
            context.run_metadata.status = RunStatus.FAILED
            await self._hooks.emit(HookEvent.ON_ERROR, error=exc, context=context)
            logger.exception("Pipeline failed in stage")
            raise

        # Quality gates
        snapshot = self._metrics.snapshot()
        # Enrich snapshot with context reports
        if context.dedup_report:
            snapshot.deduplication_report = context.dedup_report
        if context.diversity_score:
            snapshot.diversity_score = context.diversity_score

        failed = self._gates.check_all(snapshot, self._budget.max_cost)
        if failed:
            context.run_metadata.status = RunStatus.GATE_FAILED
            raise QualityGateFailedError(failed)

        context.run_metadata.status = RunStatus.COMPLETED
        context.run_metadata.completed_at = datetime.utcnow()
        await self._hooks.emit(HookEvent.PIPELINE_END, context=context)

        return PipelineResult(
            run_metadata=context.run_metadata,
            metrics=snapshot,
            samples=context.samples,
            output_dir=self._settings.output_dir,
        )

    # ------------------------------------------------------------------ #
    # Dry run
    # ------------------------------------------------------------------ #

    async def dry_run(self, topic: str) -> DryRunPlan:
        """Estimate scope and cost without making any API calls."""
        from forge.templates import TEMPLATES

        template = TEMPLATES.get(topic)
        if template:
            tmpl = template()
            pages = tmpl.estimated_page_count()
            samples = tmpl.estimated_sample_count()
        else:
            pages = self._settings.collect.max_pages
            samples = self._settings.generate.max_samples

        est = self._budget.estimate_stage("generate", samples, self._settings.default_model)
        collect_est = self._budget.estimate_stage("collect", 10, self._settings.default_model)

        return DryRunPlan(
            topic=topic,
            template=topic if topic in TEMPLATES else "generic",
            format=self._settings.generate.default_format,
            estimated_pages=pages,
            estimated_documents=min(pages, self._settings.collect.max_documents),
            estimated_samples=samples,
            estimated_tokens=est.estimated_tokens + collect_est.estimated_tokens,
            estimated_cost=est.estimated_cost + collect_est.estimated_cost,
            estimated_runtime=f"~{max(1, (pages * 2 + samples * 3) // 60)} minutes",
        )

    # ------------------------------------------------------------------ #
    # Single stage
    # ------------------------------------------------------------------ #

    async def run_stage(self, stage_name: str, topic: str) -> PipelineContext:
        """Run a single named stage."""
        context = PipelineContext(
            topic=topic,
            settings=self._settings,
            storage=self._storage,
            metrics=self._metrics,
            budget=self._budget,
        )
        for stage in self._stages:
            if stage.name == stage_name:
                return await stage.run(context)
        raise ValueError(f"Unknown stage: {stage_name}")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _setup_default_stages(self) -> None:
        """Register the built-in stages in pipeline order."""
        from forge.stages.collect import CollectStage
        from forge.stages.clean import CleanStage
        from forge.stages.analyze import AnalyzeStage
        from forge.stages.generate import GenerateStage
        from forge.stages.verify import VerifyStage
        from forge.stages.export import ExportStage

        for cls in (CollectStage, CleanStage, AnalyzeStage, GenerateStage, VerifyStage, ExportStage):
            self.add_stage(cls())

    async def _checkpoint(self, stage_name: str, context: PipelineContext) -> None:
        """Persist run state for resumability."""
        state = {
            "run_id": context.run_id,
            "topic": context.topic,
            "last_stage": stage_name,
            "stages_completed": context.run_metadata.stages_completed,
            "documents_count": len(context.documents),
            "samples_count": len(context.samples),
        }
        await self._storage.save_state(context.run_id, state)
