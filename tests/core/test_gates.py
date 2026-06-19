"""Tests for core.gates — quality gate checks."""

from forge.core.config import QualityGateSettings
from forge.core.gates import QualityGate
from forge.core.models import (
    CostReport,
    DatasetMetrics,
    DeduplicationReport,
    DiversityScore,
    VerificationReport,
)


def _make_gate(
    max_dup: float = 0.10,
    min_div: float = 0.5,
    min_ver: float = 0.7,
) -> QualityGate:
    return QualityGate(QualityGateSettings(
        max_duplicate_rate=max_dup,
        min_diversity_score=min_div,
        min_verification_score=min_ver,
    ))


def test_duplicates_pass():
    gate = _make_gate()
    report = DeduplicationReport(total_processed=100, exact_duplicates=5, near_duplicates=3)
    result = gate.check_duplicates(report)
    assert result.passed is True


def test_duplicates_fail():
    gate = _make_gate(max_dup=0.05)
    report = DeduplicationReport(total_processed=100, exact_duplicates=5, near_duplicates=3)
    result = gate.check_duplicates(report)
    assert result.passed is False
    assert result.actual_value > 0.05


def test_diversity_pass():
    gate = _make_gate()
    score = DiversityScore(overall=0.75)
    result = gate.check_diversity(score)
    assert result.passed is True


def test_diversity_fail():
    gate = _make_gate(min_div=0.8)
    score = DiversityScore(overall=0.5)
    result = gate.check_diversity(score)
    assert result.passed is False


def test_verification_pass():
    gate = _make_gate()
    report = VerificationReport(total_verified=100, passed=80, pass_rate=0.8)
    result = gate.check_verification(report)
    assert result.passed is True


def test_verification_fail():
    gate = _make_gate(min_ver=0.9)
    report = VerificationReport(total_verified=100, passed=80, pass_rate=0.8)
    result = gate.check_verification(report)
    assert result.passed is False


def test_budget_pass_no_limit():
    gate = _make_gate()
    report = CostReport(total_cost=999.0)
    result = gate.check_budget(report, limit=None)
    assert result.passed is True


def test_budget_fail():
    gate = _make_gate()
    report = CostReport(total_cost=5.50)
    result = gate.check_budget(report, limit=5.0)
    assert result.passed is False


def test_check_all_passes():
    """All gates pass with good metrics."""
    gate = _make_gate()
    metrics = DatasetMetrics(
        deduplication_report=DeduplicationReport(total_processed=100, exact_duplicates=2),
        diversity_score=DiversityScore(overall=0.8),
        verification_report=VerificationReport(pass_rate=0.9),
        cost_report=CostReport(total_cost=1.0),
    )
    failed = gate.check_all(metrics, budget_limit=10.0)
    assert failed == []


def test_check_all_fails():
    """Multiple gates should fail with bad metrics."""
    gate = _make_gate()
    metrics = DatasetMetrics(
        deduplication_report=DeduplicationReport(total_processed=10, exact_duplicates=5),
        diversity_score=DiversityScore(overall=0.2),
        verification_report=VerificationReport(pass_rate=0.3),
        cost_report=CostReport(total_cost=1.0),
    )
    failed = gate.check_all(metrics, budget_limit=10.0)
    assert len(failed) == 3  # duplicates, diversity, verification all fail
