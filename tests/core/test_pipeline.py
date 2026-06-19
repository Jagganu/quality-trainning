"""Tests for core.pipeline — pipeline construction and dry-run."""

import pytest

from forge.core.config import ForgeSettings
from forge.core.pipeline import Pipeline


def test_pipeline_construction(settings):
    """Pipeline should instantiate with default stages."""
    pipeline = Pipeline(settings)
    # Should have the default stages loaded
    assert len(pipeline._stages) >= 4  # collect, clean, analyze, generate, verify, export


@pytest.mark.asyncio
async def test_dry_run(settings):
    """dry_run should return a DryRunPlan without making API calls."""
    pipeline = Pipeline(settings)
    plan = await pipeline.dry_run("cybersecurity")

    assert plan.topic == "cybersecurity"
    assert plan.estimated_pages > 0
    assert plan.estimated_samples > 0
    assert plan.estimated_cost >= 0
    assert plan.estimated_runtime != ""
