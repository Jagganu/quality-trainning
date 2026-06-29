"""Benchmark stage — computes final dataset quality metrics (no LLM calls)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from forge.core.context import PipelineContext
from forge.core.models import SampleVerdict, VerificationReport
from forge.core.stage import Stage
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkStage(Stage):
    """Stage 6: Compute and report final dataset quality metrics."""

    @property
    def name(self) -> str:
        return "benchmark"

    async def run(self, context: PipelineContext) -> PipelineContext:
        samples = context.samples
        total = len(samples)

        if not total:
            logger.warning("No samples to benchmark")
            return context

        # Verification summary
        accepted = sum(
            1 for s in samples
            if s.verification and s.verification.verdict == SampleVerdict.ACCEPT
        )
        rejected = sum(
            1 for s in samples
            if s.verification and s.verification.verdict == SampleVerdict.REJECT
        )
        pass_rate = accepted / total if total else 0.0

        # Collect failure reasons
        failure_reasons: dict[str, int] = {}
        for s in samples:
            if s.verification and s.verification.verdict != SampleVerdict.ACCEPT:
                for issue in s.verification.issues:
                    short = issue[:80]
                    failure_reasons[short] = failure_reasons.get(short, 0) + 1

        verification_report = VerificationReport(
            total_verified=total,
            passed=accepted,
            failed=rejected,
            pass_rate=round(pass_rate, 4),
            failure_reasons=failure_reasons,
        )

        # Quality scores
        scores = [s.quality_score for s in samples if s.quality_score is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Build summary
        summary = {
            "run_id": context.run_id,
            "topic": context.topic,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "total_samples": total,
            "accepted": accepted,
            "rejected": rejected,
            "pass_rate": round(pass_rate, 4),
            "avg_quality_score": round(avg_score, 4),
            "diversity": context.diversity_score.model_dump() if context.diversity_score else None,
            "deduplication": context.dedup_report.model_dump() if context.dedup_report else None,
            "verification": verification_report.model_dump(),
            "cost": context.budget.current_cost,
        }

        # Persist summary
        await context.storage.save_document(
            "benchmarks", context.run_id, summary,
        )

        logger.info(
            "Benchmark: %d samples, %d accepted (%.0f%%), avg score %.3f, cost $%.4f",
            total, accepted, pass_rate * 100, avg_score, context.budget.current_cost,
        )

        return context
