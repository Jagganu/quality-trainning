"""Tests for forge.verification.consensus.ConsensusEngine.evaluate_with_judges
— the merged path combining cheap critics, self-consistency agreement, and
judge-ensemble verdicts. Complements test_consensus.py, which covers the
original single-critic evaluate() path (left untouched and still passing).
"""

from __future__ import annotations

import pytest

from forge.verification.consensus import ConsensusEngine
from forge.verification.models import Critique, JudgeVerdict


def _critique(*, passed: bool = True, severity: str = "none") -> Critique:
    return Critique(sample_id="s1", critic_model="format_check", passed=passed, severity=severity)


def _judge(
    verdict: str = "accept", severity: str = "none", confidence: float = 0.9,
) -> JudgeVerdict:
    return JudgeVerdict(
        sample_id="s1", judge_model="m",
        verdict=verdict, severity=severity, confidence=confidence,
    )


class TestAccept:
    def test_judges_accept_high_agreement_critics_pass(self):
        engine = ConsensusEngine()
        result = engine.evaluate_with_judges(
            "s1",
            critiques=[_critique(passed=True)],
            judge_verdicts=[_judge(confidence=0.9), _judge(confidence=0.8)],
            agreement_ratio=1.0,
        )
        assert result.final_verdict == "accept"
        assert result.confidence == pytest.approx(0.8)  # min(judge_conf=0.8, agreement=1.0)

    def test_agreement_exactly_at_min_threshold_accepts(self):
        engine = ConsensusEngine()
        result = engine.evaluate_with_judges(
            "s1", critiques=[], judge_verdicts=[_judge(confidence=0.9)],
            agreement_ratio=0.34, min_agreement=0.34,
        )
        assert result.final_verdict == "accept"


class TestReject:
    def test_fatal_critique_rejects_even_if_judges_accept(self):
        engine = ConsensusEngine()
        result = engine.evaluate_with_judges(
            "s1",
            critiques=[_critique(passed=False, severity="fatal")],
            judge_verdicts=[_judge(confidence=0.99), _judge(confidence=0.99)],
            agreement_ratio=1.0,
        )
        assert result.final_verdict == "reject"

    def test_judge_ensemble_reject_propagates(self):
        engine = ConsensusEngine()
        result = engine.evaluate_with_judges(
            "s1", critiques=[],
            judge_verdicts=[_judge(verdict="reject"), _judge(verdict="accept")],
            agreement_ratio=1.0,
        )
        assert result.final_verdict == "reject"

    def test_judge_fatal_severity_rejects(self):
        engine = ConsensusEngine()
        result = engine.evaluate_with_judges(
            "s1", critiques=[],
            judge_verdicts=[_judge(severity="fatal"), _judge(confidence=0.99)],
            agreement_ratio=1.0,
        )
        assert result.final_verdict == "reject"
        assert result.confidence == 0.0


class TestRevise:
    def test_low_agreement_below_threshold_revises_even_if_judges_accept(self):
        engine = ConsensusEngine()
        result = engine.evaluate_with_judges(
            "s1", critiques=[],
            judge_verdicts=[_judge(confidence=0.9), _judge(confidence=0.9)],
            agreement_ratio=0.2, min_agreement=0.34,
        )
        assert result.final_verdict == "revise"

    def test_no_judge_verdicts_defaults_to_revise(self):
        """An empty judge list (e.g. ensemble call failed entirely)
        should never silently resolve to accept.
        """
        engine = ConsensusEngine()
        result = engine.evaluate_with_judges(
            "s1", critiques=[], judge_verdicts=[], agreement_ratio=1.0,
        )
        assert result.final_verdict == "revise"
        assert result.confidence == 0.0

    def test_failed_critique_check_revises(self):
        engine = ConsensusEngine()
        result = engine.evaluate_with_judges(
            "s1",
            critiques=[_critique(passed=False, severity="minor")],
            judge_verdicts=[_judge(confidence=0.9)],
            agreement_ratio=1.0,
        )
        assert result.final_verdict == "revise"


class TestResultStructure:
    def test_judge_verdicts_stored_on_result(self):
        engine = ConsensusEngine()
        verdicts = [_judge(confidence=0.9), _judge(confidence=0.8)]
        result = engine.evaluate_with_judges(
            "s1", critiques=[], judge_verdicts=verdicts, agreement_ratio=1.0,
        )
        assert len(result.judge_verdicts) == 2

    def test_agreement_ratio_stored_on_result(self):
        engine = ConsensusEngine()
        result = engine.evaluate_with_judges(
            "s1", critiques=[], judge_verdicts=[_judge()], agreement_ratio=0.67,
        )
        assert result.agreement_ratio == 0.67

    def test_original_evaluate_path_unaffected(self):
        """evaluate_with_judges is additive — the original evaluate()
        path's defaults for the new fields should stay neutral.
        """
        engine = ConsensusEngine()
        result = engine.evaluate("s1", critiques=[], scores=[])
        assert result.judge_verdicts == []
        assert result.agreement_ratio == 0.0
