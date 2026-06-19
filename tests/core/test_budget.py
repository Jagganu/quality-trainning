"""Tests for core.budget — cost tracking and enforcement."""

import pytest

from forge.core.budget import BudgetExceededError, CostBudget


def test_unlimited_budget():
    """Unlimited budget should never raise."""
    b = CostBudget(max_cost=None)
    b.record_usage(1000, 500, "gpt-4o-mini", 0.05, "generate")
    b.record_usage(1000, 500, "gpt-4o-mini", 0.05, "verify")
    assert b.current_cost == pytest.approx(0.10, abs=1e-6)
    assert b.remaining() is None


def test_budget_tracking():
    """Cost should accumulate by model and stage."""
    b = CostBudget(max_cost=1.0)
    b.record_usage(100, 200, "gpt-4o-mini", 0.01, "collect")
    b.record_usage(200, 400, "gpt-4o", 0.05, "generate")

    report = b.report()
    assert report.total_cost == pytest.approx(0.06, abs=1e-6)
    assert report.cost_by_model["gpt-4o-mini"] == pytest.approx(0.01)
    assert report.cost_by_stage["generate"] == pytest.approx(0.05)
    assert report.total_tokens_in == 300
    assert report.total_tokens_out == 600


def test_budget_exceeded():
    """Should raise BudgetExceededError when limit is hit."""
    b = CostBudget(max_cost=0.05)
    b.record_usage(100, 100, "gpt-4o-mini", 0.03, "collect")
    with pytest.raises(BudgetExceededError) as exc_info:
        b.record_usage(100, 100, "gpt-4o-mini", 0.03, "generate")
    assert exc_info.value.spent == pytest.approx(0.06, abs=1e-6)
    assert exc_info.value.limit == 0.05


def test_remaining_budget():
    """remaining() should report correct values."""
    b = CostBudget(max_cost=1.0)
    assert b.remaining() == pytest.approx(1.0)
    b.record_usage(100, 100, "gpt-4o-mini", 0.25, "collect")
    assert b.remaining() == pytest.approx(0.75, abs=1e-6)


def test_estimate_stage():
    """estimate_stage should return a CostEstimate."""
    b = CostBudget(max_cost=10.0)
    est = b.estimate_stage("generate", 10, "gpt-4o-mini")
    assert est.estimated_tokens > 0
    assert est.estimated_cost > 0
    assert est.model == "gpt-4o-mini"
