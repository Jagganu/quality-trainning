"""Tests for forge.verification.scorers — LLMScorer and Scorer ABC.

These tests focus on the unit-testable aspects of the scorer:
  - Initialization and weight configuration
  - Graceful degradation on API failure
  - Correct weighted-average calculation (via mocked LLM responses)
  - Abstract base class contract
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from forge.core.models import Sample, SampleLineage
from forge.verification.models import ScoreResult
from forge.verification.scorers import LLMScorer, Scorer


# ── Helpers ───────────────────────────────────────────────────────────────

def _sample(content: dict | None = None) -> Sample:
    """Build a minimal Sample for testing."""
    return Sample(
        lineage=SampleLineage(
            pipeline_run_id="run-test",
            template="testing",
            format="reasoning",
            generation_model="test-model",
        ),
        content=content or {
            "question": "What is SQL injection?",
            "analysis": "SQL injection is a code injection technique...",
            "answer": "It exploits improperly sanitized user input...",
            "metadata": {"difficulty": "beginner"},
        },
    )


def _mock_llm_response(json_text: str):
    """Build a mock litellm response object."""
    message = SimpleNamespace(content=json_text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


# ── Scorer ABC ────────────────────────────────────────────────────────────

class TestScorerABC:
    """The Scorer abstract class should enforce the contract."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Scorer()  # type: ignore[abstract]

    def test_subclass_must_implement_score(self):
        """A subclass without score() should fail to instantiate."""
        class BadScorer(Scorer):
            pass

        with pytest.raises(TypeError):
            BadScorer()  # type: ignore[abstract]


# ── LLMScorer initialization ─────────────────────────────────────────────

class TestLLMScorerInit:
    """LLMScorer should accept default and custom configurations."""

    def test_default_weights(self):
        scorer = LLMScorer()
        assert scorer.weights == {"accuracy": 0.4, "relevance": 0.3, "completeness": 0.3}

    def test_default_model(self):
        scorer = LLMScorer()
        assert scorer.model == "gpt-4o-mini"

    def test_custom_model(self):
        scorer = LLMScorer(model="claude-3-haiku-20240307")
        assert scorer.model == "claude-3-haiku-20240307"

    def test_custom_weights(self):
        custom = {"accuracy": 0.8, "relevance": 0.1, "completeness": 0.1}
        scorer = LLMScorer(weights=custom)
        assert scorer.weights == custom

    def test_weights_are_independent_copy(self):
        """Modifying the scorer's weights should not affect the default."""
        s1 = LLMScorer()
        s2 = LLMScorer()
        s1.weights["accuracy"] = 0.99
        assert s2.weights["accuracy"] == 0.4


# ── LLMScorer.score with mocked LLM ──────────────────────────────────────

class TestLLMScorerScore:
    """Test the score method with a mocked litellm.acompletion."""

    @pytest.mark.asyncio
    async def test_successful_scoring(self):
        """A clean JSON response should produce correct weighted scores."""
        mock_response = _mock_llm_response(
            '{"accuracy": 0.9, "relevance": 0.8, "completeness": 0.7}'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            scorer = LLMScorer()
            result = await scorer.score(_sample())

        assert isinstance(result, ScoreResult)
        assert result.accuracy == 0.9
        assert result.relevance == 0.8
        assert result.completeness == 0.7
        # overall = 0.4*0.9 + 0.3*0.8 + 0.3*0.7 = 0.36 + 0.24 + 0.21 = 0.81
        assert result.overall == pytest.approx(0.81, abs=1e-4)

    @pytest.mark.asyncio
    async def test_markdown_wrapped_json(self):
        """The scorer should strip markdown code fences from the response."""
        mock_response = _mock_llm_response(
            '```json\n{"accuracy": 0.5, "relevance": 0.5, "completeness": 0.5}\n```'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            scorer = LLMScorer()
            result = await scorer.score(_sample())

        assert result.accuracy == 0.5
        assert result.overall == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_custom_weights_affect_overall(self):
        """Custom weights should change the overall calculation."""
        mock_response = _mock_llm_response(
            '{"accuracy": 1.0, "relevance": 0.0, "completeness": 0.0}'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            scorer = LLMScorer(weights={"accuracy": 1.0, "relevance": 0.0, "completeness": 0.0})
            result = await scorer.score(_sample())

        # With 100% weight on accuracy=1.0 → overall=1.0
        assert result.overall == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_sample_id_propagated(self):
        """The result should carry the correct sample_id from lineage."""
        sample = _sample()
        expected_id = sample.lineage.sample_id

        mock_response = _mock_llm_response('{"accuracy": 0.5, "relevance": 0.5, "completeness": 0.5}')
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await LLMScorer().score(sample)

        assert result.sample_id == expected_id

    @pytest.mark.asyncio
    async def test_scorer_model_propagated(self):
        """The result should carry the scorer's model name."""
        mock_response = _mock_llm_response('{"accuracy": 0.5, "relevance": 0.5, "completeness": 0.5}')
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await LLMScorer(model="gpt-4o").score(_sample())

        assert result.scorer_model == "gpt-4o"


# ── LLMScorer error handling ─────────────────────────────────────────────

class TestLLMScorerErrors:
    """The scorer should degrade gracefully on failures."""

    @pytest.mark.asyncio
    async def test_api_error_returns_zero_scores(self):
        """An API exception should return a ScoreResult with all zeros."""
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("API down")):
            scorer = LLMScorer()
            result = await scorer.score(_sample())

        # On failure, ScoreResult defaults: accuracy=0.0, overall=0.0
        assert result.accuracy == 0.0
        assert result.relevance == 0.0
        assert result.completeness == 0.0
        assert result.overall == 0.0

    @pytest.mark.asyncio
    async def test_invalid_json_returns_zero_scores(self):
        """Unparseable JSON from the LLM should not crash."""
        mock_response = _mock_llm_response("This is not JSON at all!")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await LLMScorer().score(_sample())

        assert result.overall == 0.0

    @pytest.mark.asyncio
    async def test_null_content_returns_zero_scores(self):
        """A response with null content should be handled."""
        message = SimpleNamespace(content=None)
        choice = SimpleNamespace(message=message)
        mock_response = SimpleNamespace(choices=[choice])

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await LLMScorer().score(_sample())

        # None content → "{}" fallback → all 0.0
        assert result.overall == 0.0

    @pytest.mark.asyncio
    async def test_missing_keys_default_to_zero(self):
        """If the LLM omits a dimension, it should default to 0.0."""
        mock_response = _mock_llm_response('{"accuracy": 0.9}')
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await LLMScorer().score(_sample())

        assert result.accuracy == 0.9
        assert result.relevance == 0.0
        assert result.completeness == 0.0
        # overall = 0.4*0.9 + 0.3*0 + 0.3*0 = 0.36
        assert result.overall == pytest.approx(0.36, abs=1e-4)

    @pytest.mark.asyncio
    async def test_error_preserves_sample_id(self):
        """Even on failure, the sample_id should be correct."""
        sample = _sample()
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("fail")):
            result = await LLMScorer().score(sample)

        assert result.sample_id == sample.lineage.sample_id
        assert result.scorer_model == "gpt-4o-mini"
