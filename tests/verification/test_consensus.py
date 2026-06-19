"""Tests for forge.verification.consensus — ConsensusEngine verdict logic.

The ConsensusEngine uses three rules in priority order:
  1. Fatal severity → always reject.
  2. pass_rate ≥ min_pass_rate AND avg_score ≥ min_score → accept.
  3. Everything else → revise.

Confidence is computed as: round(avg_score × pass_rate, 4).
"""

from __future__ import annotations

import pytest

from forge.verification.consensus import ConsensusEngine
from forge.verification.models import ConsensusResult, Critique, ScoreResult


# ── Helpers ───────────────────────────────────────────────────────────────

def _critique(*, passed: bool = True, severity: str = "none", issues: list[str] | None = None) -> Critique:
    """Shorthand for building a Critique."""
    return Critique(
        sample_id="s1",
        critic_model="test-critic",
        passed=passed,
        severity=severity,
        issues=issues or [],
    )


def _score(overall: float = 0.8) -> ScoreResult:
    """Shorthand for building a ScoreResult."""
    return ScoreResult(
        sample_id="s1",
        scorer_model="test-scorer",
        accuracy=overall,
        relevance=overall,
        completeness=overall,
        overall=overall,
    )


# ── Accept verdicts ──────────────────────────────────────────────────────

class TestAcceptVerdicts:
    """Samples that pass all critics and have scores above threshold."""

    def test_all_pass_high_score(self):
        engine = ConsensusEngine(min_pass_rate=0.6, min_score=0.7)
        result = engine.evaluate(
            "s1",
            critiques=[_critique(passed=True), _critique(passed=True)],
            scores=[_score(0.9)],
        )
        assert result.final_verdict == "accept"
        # confidence = 0.9 * 1.0 = 0.9
        assert result.confidence == pytest.approx(0.9)

    def test_exactly_at_thresholds(self):
        """Thresholds are inclusive: exactly min_pass_rate and min_score → accept."""
        engine = ConsensusEngine(min_pass_rate=0.5, min_score=0.7)
        result = engine.evaluate(
            "s1",
            critiques=[_critique(passed=True), _critique(passed=False)],  # 50% pass
            scores=[_score(0.7)],  # exactly min_score
        )
        assert result.final_verdict == "accept"

    def test_no_critiques_means_100_pass_rate(self):
        """With no critiques, pass_rate defaults to 1.0."""
        engine = ConsensusEngine(min_pass_rate=0.6, min_score=0.7)
        result = engine.evaluate("s1", critiques=[], scores=[_score(0.8)])
        assert result.final_verdict == "accept"
        # confidence = 0.8 * 1.0 = 0.8
        assert result.confidence == pytest.approx(0.8)

    def test_reasoning_mentions_rates(self):
        """Accept reasoning should include the pass rate and score values."""
        engine = ConsensusEngine()
        result = engine.evaluate(
            "s1",
            critiques=[_critique(passed=True)],
            scores=[_score(0.85)],
        )
        assert "100%" in result.reasoning or "pass rate" in result.reasoning.lower()


# ── Reject verdicts ──────────────────────────────────────────────────────

class TestRejectVerdicts:
    """Samples with fatal critiques are always rejected."""

    def test_fatal_severity_rejects(self):
        """A single fatal critique should override everything."""
        engine = ConsensusEngine()
        result = engine.evaluate(
            "s1",
            critiques=[
                _critique(passed=True),
                _critique(passed=False, severity="fatal", issues=["plagiarised"]),
            ],
            scores=[_score(0.95)],
        )
        assert result.final_verdict == "reject"
        assert "fatal" in result.reasoning.lower()

    def test_fatal_even_with_all_pass(self):
        """Fatal rejects even when all critiques otherwise pass."""
        engine = ConsensusEngine()
        result = engine.evaluate(
            "s1",
            critiques=[
                _critique(passed=True, severity="fatal"),  # passed=True but severity=fatal
            ],
            scores=[_score(0.99)],
        )
        assert result.final_verdict == "reject"


# ── Revise verdicts ──────────────────────────────────────────────────────

class TestReviseVerdicts:
    """Samples that are neither accepted nor fatally rejected."""

    def test_low_pass_rate(self):
        """Below min_pass_rate with adequate score → revise."""
        engine = ConsensusEngine(min_pass_rate=0.8, min_score=0.5)
        result = engine.evaluate(
            "s1",
            critiques=[_critique(passed=False), _critique(passed=True)],  # 50%
            scores=[_score(0.9)],
        )
        assert result.final_verdict == "revise"
        assert "pass rate" in result.reasoning.lower()

    def test_low_score(self):
        """Below min_score with adequate pass rate → revise."""
        engine = ConsensusEngine(min_pass_rate=0.5, min_score=0.8)
        result = engine.evaluate(
            "s1",
            critiques=[_critique(passed=True)],
            scores=[_score(0.6)],
        )
        assert result.final_verdict == "revise"
        assert "score" in result.reasoning.lower()

    def test_both_low(self):
        """Both pass rate AND score below thresholds → revise (not reject)."""
        engine = ConsensusEngine(min_pass_rate=0.8, min_score=0.8)
        result = engine.evaluate(
            "s1",
            critiques=[_critique(passed=False), _critique(passed=True)],  # 50%
            scores=[_score(0.5)],
        )
        assert result.final_verdict == "revise"
        assert "pass rate" in result.reasoning.lower()
        assert "score" in result.reasoning.lower()

    def test_no_scores_means_zero_avg(self):
        """With no scores, avg_score = 0.0, which is below any min_score."""
        engine = ConsensusEngine(min_pass_rate=0.5, min_score=0.1)
        result = engine.evaluate(
            "s1",
            critiques=[_critique(passed=True)],
            scores=[],  # avg_score = 0.0
        )
        assert result.final_verdict == "revise"


# ── Confidence calculation ───────────────────────────────────────────────

class TestConfidence:
    """confidence = round(avg_score × pass_rate, 4)."""

    def test_simple_calculation(self):
        engine = ConsensusEngine()
        result = engine.evaluate(
            "s1",
            critiques=[_critique(passed=True), _critique(passed=False)],  # 50% pass
            scores=[_score(0.8)],
        )
        # 0.8 * 0.5 = 0.4
        assert result.confidence == pytest.approx(0.4)

    def test_full_confidence(self):
        engine = ConsensusEngine()
        result = engine.evaluate(
            "s1",
            critiques=[_critique(passed=True)],
            scores=[_score(1.0)],
        )
        assert result.confidence == pytest.approx(1.0)

    def test_zero_confidence_no_scores(self):
        """No scores → avg_score=0 → confidence=0."""
        engine = ConsensusEngine()
        result = engine.evaluate("s1", critiques=[], scores=[])
        assert result.confidence == 0.0

    def test_multiple_scores_averaged(self):
        """Average of multiple ScoreResults is used."""
        engine = ConsensusEngine()
        result = engine.evaluate(
            "s1",
            critiques=[_critique(passed=True)],
            scores=[_score(0.6), _score(1.0)],  # avg = 0.8
        )
        # 0.8 * 1.0 = 0.8
        assert result.confidence == pytest.approx(0.8)

    def test_rounded_to_4_places(self):
        engine = ConsensusEngine()
        result = engine.evaluate(
            "s1",
            critiques=[_critique(passed=True), _critique(passed=True), _critique(passed=False)],
            scores=[_score(0.777)],
        )
        # pass_rate = 2/3 ≈ 0.6667, avg_score = 0.777
        # 0.777 * 0.6667 ≈ 0.5180
        assert result.confidence == round(result.confidence, 4)


# ── Result structure ─────────────────────────────────────────────────────

class TestResultStructure:
    """The ConsensusResult should carry all inputs through."""

    def test_sample_id_preserved(self):
        engine = ConsensusEngine()
        result = engine.evaluate("my-sample-42", critiques=[], scores=[])
        assert result.sample_id == "my-sample-42"

    def test_critiques_stored(self):
        cs = [_critique(), _critique(passed=False)]
        engine = ConsensusEngine()
        result = engine.evaluate("s1", critiques=cs, scores=[])
        assert len(result.critiques) == 2

    def test_scores_stored(self):
        ss = [_score(0.7), _score(0.9)]
        engine = ConsensusEngine()
        result = engine.evaluate("s1", critiques=[], scores=ss)
        assert len(result.scores) == 2

    def test_returns_consensus_result_type(self):
        engine = ConsensusEngine()
        result = engine.evaluate("s1", critiques=[], scores=[])
        assert isinstance(result, ConsensusResult)


# ── Engine configuration ─────────────────────────────────────────────────

class TestEngineConfiguration:
    """ConsensusEngine accepts custom thresholds."""

    def test_default_thresholds(self):
        engine = ConsensusEngine()
        assert engine.min_pass_rate == 0.6
        assert engine.min_score == 0.7

    def test_custom_thresholds(self):
        engine = ConsensusEngine(min_pass_rate=0.9, min_score=0.95)
        assert engine.min_pass_rate == 0.9
        assert engine.min_score == 0.95

    def test_strict_engine_rejects_moderate_quality(self):
        """A very strict engine should revise what a lenient one accepts."""
        strict = ConsensusEngine(min_pass_rate=1.0, min_score=0.95)
        lenient = ConsensusEngine(min_pass_rate=0.1, min_score=0.1)

        critiques = [_critique(passed=True), _critique(passed=False)]
        scores = [_score(0.8)]

        assert strict.evaluate("s1", critiques, scores).final_verdict == "revise"
        assert lenient.evaluate("s1", critiques, scores).final_verdict == "accept"
