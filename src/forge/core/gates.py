"""Quality gates — configurable thresholds that fail the pipeline on low quality."""

from __future__ import annotations

from forge.core.config import QualityGateSettings
from forge.core.models import (
    CostReport,
    DatasetMetrics,
    DeduplicationReport,
    DiversityScore,
    GateResult,
    VerificationReport,
)
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class QualityGateFailedError(Exception):
    """Raised when one or more quality gates are violated."""

    def __init__(self, failed_gates: list[GateResult]) -> None:
        self.failed_gates = failed_gates
        names = ", ".join(g.gate for g in failed_gates)
        super().__init__(f"Quality gates failed: {names}")


class QualityGate:
    """Checks dataset quality against configurable thresholds."""

    def __init__(self, config: QualityGateSettings) -> None:
        self._config = config

    def check_duplicates(self, report: DeduplicationReport) -> GateResult:
        """Fail if duplicate rate exceeds threshold."""
        if report.total_processed == 0:
            return GateResult(gate="duplicates", passed=True, actual_value=0.0,
                              threshold=self._config.max_duplicate_rate, message="No documents to check")
        rate = (report.exact_duplicates + report.near_duplicates) / report.total_processed
        passed = rate <= self._config.max_duplicate_rate
        return GateResult(
            gate="duplicates", passed=passed, actual_value=round(rate, 4),
            threshold=self._config.max_duplicate_rate,
            message="" if passed else f"Duplicate rate {rate:.1%} exceeds {self._config.max_duplicate_rate:.1%}",
        )

    def check_diversity(self, score: DiversityScore) -> GateResult:
        """Fail if diversity score is below threshold."""
        passed = score.overall >= self._config.min_diversity_score
        return GateResult(
            gate="diversity", passed=passed, actual_value=round(score.overall, 4),
            threshold=self._config.min_diversity_score,
            message="" if passed else f"Diversity {score.overall:.2f} below {self._config.min_diversity_score:.2f}",
        )

    def check_verification(self, report: VerificationReport) -> GateResult:
        """Fail if verification pass rate is below threshold."""
        passed = report.pass_rate >= self._config.min_verification_score
        return GateResult(
            gate="verification", passed=passed, actual_value=round(report.pass_rate, 4),
            threshold=self._config.min_verification_score,
            message="" if passed else f"Verification {report.pass_rate:.1%} below {self._config.min_verification_score:.1%}",
        )

    def check_budget(self, report: CostReport, limit: float | None) -> GateResult:
        """Fail if budget was exceeded."""
        if limit is None:
            return GateResult(gate="budget", passed=True, actual_value=report.total_cost,
                              threshold=0.0, message="No budget limit set")
        passed = report.total_cost <= limit
        return GateResult(
            gate="budget", passed=passed, actual_value=round(report.total_cost, 4),
            threshold=limit,
            message="" if passed else f"Cost ${report.total_cost:.2f} exceeds ${limit:.2f}",
        )

    def check_all(self, metrics: DatasetMetrics, budget_limit: float | None = None) -> list[GateResult]:
        """Run all gates. Returns only *failed* gates (empty = all passed)."""
        results = [
            self.check_duplicates(metrics.deduplication_report),
            self.check_diversity(metrics.diversity_score),
            self.check_verification(metrics.verification_report),
            self.check_budget(metrics.cost_report, budget_limit),
        ]
        failed = [r for r in results if not r.passed]
        for f in failed:
            logger.warning("Quality gate FAILED: %s — %s", f.gate, f.message)
        return failed
