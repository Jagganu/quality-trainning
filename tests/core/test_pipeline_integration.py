"""Integration tests for the new self-consistency / judge-ensemble / refine
re-verify wiring, exercising real Stage.run() calls with mocked litellm
responses (no live API access required).

These complement the unit tests in tests/stages/generate/test_self_consistency.py
and tests/verification/test_*.py, which test the logic in isolation — these
confirm the stages actually wire that logic together correctly end to end.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from forge.core.budget import CostBudget
from forge.core.config import ForgeSettings
from forge.core.context import PipelineContext
from forge.core.models import (
    AnalysisResult,
    CleanedDocument,
    Document,
    Sample,
    SampleLineage,
    SampleVerdict,
    SubtopicPlan,
    VerificationResult,
)
from forge.metrics.collector import MetricsCollector
from forge.stages.generate.generator import GenerateStage
from forge.stages.refine.refiner import RefineStage
from forge.stages.verify.verifier import VerifyStage
from forge.storage.filesystem import FilesystemStorage


def _mock_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=80),
    )


@pytest.fixture
def context(tmp_path):
    def _make(settings: ForgeSettings) -> PipelineContext:
        ctx = PipelineContext(
            topic="binary search algorithms",
            settings=settings,
            storage=FilesystemStorage(tmp_path),
            metrics=MetricsCollector(),
            budget=CostBudget(),
        )
        ctx.documents = [
            Document(doc_id="doc1", url="https://example.com/bs", title="Binary Search",
                      content=(
                          "Binary search is a search algorithm that finds a target value "
                          "within a sorted array. It works by repeatedly halving the search "
                          "space: comparing the target to the middle element and eliminating "
                          "half of the remaining elements at each comparison. Because the "
                          "search space shrinks by half every step, binary search runs in "
                          "O(log n) time, much faster than a linear O(n) scan for large arrays."
                      )),
        ]
        ctx.cleaned_documents = [
            CleanedDocument(doc_id="cd1", source_doc_id="doc1",
                             chunks=["Binary search halves the search space each step."]),
        ]
        ctx.analysis = AnalysisResult(
            topic="binary search algorithms",
            generation_plan=[
                SubtopicPlan(
                    name="complexity",
                    concepts=["binary search time complexity", "recursion vs iteration"],
                    sample_count=4,
                    difficulties={"intermediate": 1},
                )
            ],
        )
        ctx.template_name = "test-template"
        return ctx
    return _make


class TestGenerateSelfConsistencyIntegration:
    """GenerateStage.run() with self_consistency_n > 1."""

    @pytest.mark.asyncio
    async def test_majority_cluster_survives_to_a_sample(self, context):
        settings = ForgeSettings(default_model="mock-model")
        settings.generate.self_consistency_n = 3
        settings.generate.max_concurrent = 1
        settings.generate.batch_size = 1
        settings.generate.max_samples = 1
        ctx = context(settings)
        # Only the first concept will be reached given max_samples=1.
        ctx.analysis.generation_plan[0].concepts = ["binary search time complexity"]

        answers = ["O(log n)", "O(log n)", "O(n)"]
        call_count = {"n": 0}

        async def fake_acompletion(model, messages, **kwargs):
            i = call_count["n"]
            call_count["n"] += 1
            payload = {
                "question": "Explain binary search complexity",
                "analysis": f"chain {i}",
                "answer": answers[i],
                "metadata": {},
            }
            return _mock_response(json.dumps(payload))

        with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
            ctx = await GenerateStage().run(ctx)

        assert len(ctx.samples) == 1
        sample = ctx.samples[0]
        assert sample.content["answer"] == "O(log n)"
        assert sample.content["metadata"]["self_consistency_agreement"] == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_no_agreement_drops_the_sample(self, context):
        settings = ForgeSettings(default_model="mock-model")
        settings.generate.self_consistency_n = 3
        settings.generate.max_concurrent = 1
        settings.generate.batch_size = 1
        settings.generate.max_samples = 1
        ctx = context(settings)
        ctx.analysis.generation_plan[0].concepts = ["recursion vs iteration"]

        answers = ["Answer A", "Answer B", "Answer C"]
        call_count = {"n": 0}

        async def fake_acompletion(model, messages, **kwargs):
            i = call_count["n"]
            call_count["n"] += 1
            payload = {"question": "q", "analysis": "a", "answer": answers[i], "metadata": {}}
            return _mock_response(json.dumps(payload))

        with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
            ctx = await GenerateStage().run(ctx)

        assert len(ctx.samples) == 0
        assert ctx.metrics.counter_value("samples_invalid") == 1

    @pytest.mark.asyncio
    async def test_self_consistency_disabled_preserves_original_behavior(self, context):
        """n=1 (default) should behave exactly like the pre-upgrade pipeline."""
        settings = ForgeSettings(default_model="mock-model")
        settings.generate.batch_size = 1
        settings.generate.max_samples = 1
        ctx = context(settings)
        ctx.analysis.generation_plan[0].concepts = ["binary search time complexity"]

        async def fake_acompletion(model, messages, **kwargs):
            payload = {"question": "q", "analysis": "a", "answer": "O(log n)", "metadata": {}}
            return _mock_response(json.dumps(payload))

        with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
            ctx = await GenerateStage().run(ctx)

        assert len(ctx.samples) == 1
        # No self-consistency metadata should be injected on the single-shot path.
        assert "self_consistency_agreement" not in ctx.samples[0].content["metadata"]


class TestVerifyJudgeEnsembleIntegration:
    """VerifyStage.run() with 2+ judge_models configured."""

    @pytest.mark.asyncio
    async def test_both_judges_accept_good_sample(self, context):
        settings = ForgeSettings(default_model="mock-model")
        settings.verify.judge_models = ["judge-a", "judge-b"]
        ctx = context(settings)
        ctx.samples = [Sample(
            lineage=SampleLineage(template="t", pipeline_run_id=ctx.run_id, format="reasoning"),
            content={
                "question": "What is the time complexity of binary search?",
                "analysis": "Halves the search space each comparison.",
                "answer": "O(log n).",
                "metadata": {"self_consistency_agreement": 1.0},
            },
        )]

        async def fake_acompletion(model, messages, **kwargs):
            data = {"verdict": "accept", "severity": "none", "confidence": 0.9, "reasoning": "Good"}
            return _mock_response(json.dumps(data))

        with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
            ctx = await VerifyStage().run(ctx)

        assert ctx.samples[0].verification.verdict == SampleVerdict.ACCEPT

    @pytest.mark.asyncio
    async def test_one_fatal_judge_rejects_despite_other_accepting(self, context):
        settings = ForgeSettings(default_model="mock-model")
        settings.verify.judge_models = ["judge-a", "judge-b"]
        ctx = context(settings)
        ctx.samples = [Sample(
            lineage=SampleLineage(template="t", pipeline_run_id=ctx.run_id, format="reasoning"),
            content={
                "question": "What is binary search?",
                "analysis": "It's a sorting algorithm.",
                "answer": "O(n log n).",
                "metadata": {"self_consistency_agreement": 1.0},
            },
        )]

        async def fake_acompletion(model, messages, **kwargs):
            if model == "judge-a":
                data = {"verdict": "reject", "severity": "fatal", "confidence": 0.1,
                        "reasoning": "Conflates search with sort"}
            else:
                data = {"verdict": "accept", "severity": "none", "confidence": 0.9,
                        "reasoning": "Looks fine"}
            return _mock_response(json.dumps(data))

        with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
            ctx = await VerifyStage().run(ctx)

        assert ctx.samples[0].verification.verdict == SampleVerdict.REJECT

    @pytest.mark.asyncio
    async def test_single_critic_path_still_works_unchanged(self, context):
        """judge_models empty -> falls back to the original single-critic path."""
        settings = ForgeSettings(default_model="mock-model")
        settings.verify.critic_model = "critic-model"
        ctx = context(settings)
        ctx.samples = [Sample(
            lineage=SampleLineage(template="t", pipeline_run_id=ctx.run_id, format="reasoning"),
            content={
                "question": "What is the time complexity of binary search?",
                "analysis": "Halves the search space each comparison.",
                "answer": "O(log n).",
                "metadata": {},
            },
        )]

        async def fake_acompletion(model, messages, **kwargs):
            prompt = messages[-1]["content"]
            if "strict quality-assurance" in prompt:
                data = {"issues": [], "severity": "none", "reasoning": "fine", "passed": True}
            else:
                data = {"accuracy": 0.9, "relevance": 0.9, "completeness": 0.9, "overall": 0.9}
            return _mock_response(json.dumps(data))

        with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
            ctx = await VerifyStage().run(ctx)

        assert ctx.samples[0].verification.verdict == SampleVerdict.ACCEPT


class TestRefineReverifyIntegration:
    """RefineStage.run() actually re-verifies instead of auto-accepting."""

    @pytest.mark.asyncio
    async def test_bad_refinement_stays_in_revise_then_retries_to_accept(self, context):
        settings = ForgeSettings(default_model="mock-model")
        settings.verify.critic_model = "critic-model"
        settings.refine.max_retries = 2
        ctx = context(settings)
        ctx.samples = [Sample(
            lineage=SampleLineage(template="t", pipeline_run_id=ctx.run_id, format="reasoning"),
            content={"question": "q", "analysis": "a", "answer": "O(n)", "metadata": {}},
            verification=VerificationResult(
                verdict=SampleVerdict.REVISE, issues=["wrong complexity"],
            ),
        )]

        attempt = {"n": 0}

        async def fake_acompletion(model, messages, **kwargs):
            prompt = messages[-1]["content"]
            if "improving an AI training sample" in prompt:
                attempt["n"] += 1
                answer = "O(n), still wrong" if attempt["n"] == 1 else "O(log n)"
                return _mock_response(json.dumps(
                    {"question": "q", "analysis": "a", "answer": answer, "metadata": {}}
                ))
            if "strict quality-assurance" in prompt:
                passed = "still wrong" not in prompt
                data = {"issues": [] if passed else ["still incorrect"],
                        "severity": "none" if passed else "major",
                        "reasoning": "checked", "passed": passed}
                return _mock_response(json.dumps(data))
            # scorer
            overall = 0.9 if "still wrong" not in prompt else 0.3
            return _mock_response(json.dumps({
                "accuracy": overall, "relevance": overall,
                "completeness": overall, "overall": overall,
            }))

        with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
            ctx = await RefineStage().run(ctx)

        sample = ctx.samples[0]
        assert sample.verification.verdict == SampleVerdict.ACCEPT
        assert sample.content["answer"] == "O(log n)"
        # Proves 2 refine attempts actually happened, not a single blind accept.
        assert sample.lineage.stage_versions["refine"] == "1.0-attempt2"

    @pytest.mark.asyncio
    async def test_still_failing_after_max_retries_does_not_get_accepted(self, context):
        """The old bug: this scenario used to silently flip to ACCEPT no
        matter what. It must not anymore.
        """
        settings = ForgeSettings(default_model="mock-model")
        settings.verify.critic_model = "critic-model"
        settings.refine.max_retries = 2
        ctx = context(settings)
        ctx.samples = [Sample(
            lineage=SampleLineage(template="t", pipeline_run_id=ctx.run_id, format="reasoning"),
            content={"question": "q", "analysis": "a", "answer": "O(n)", "metadata": {}},
            verification=VerificationResult(
                verdict=SampleVerdict.REVISE, issues=["wrong complexity"],
            ),
        )]

        async def fake_acompletion(model, messages, **kwargs):
            prompt = messages[-1]["content"]
            if "improving an AI training sample" in prompt:
                return _mock_response(json.dumps({
                    "question": "q", "analysis": "a",
                    "answer": "still wrong every time", "metadata": {},
                }))
            if "strict quality-assurance" in prompt:
                return _mock_response(json.dumps({
                    "issues": ["still incorrect"], "severity": "major",
                    "reasoning": "nope", "passed": False,
                }))
            return _mock_response(json.dumps(
                {"accuracy": 0.2, "relevance": 0.2, "completeness": 0.2, "overall": 0.2}
            ))

        with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
            ctx = await RefineStage().run(ctx)

        assert ctx.samples[0].verification.verdict != SampleVerdict.ACCEPT
