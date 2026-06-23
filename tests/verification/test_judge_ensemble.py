"""Tests for forge.verification.judge_ensemble — merge_judge_verdicts and
JudgeEnsemble construction. The actual LLM-calling ``_judge``/``evaluate``
methods are exercised via mocked litellm calls; merge logic itself needs
no mocking since it's pure data transformation.
"""

from __future__ import annotations

import pytest

from forge.verification.judge_ensemble import JudgeEnsemble, merge_judge_verdicts
from forge.verification.models import JudgeVerdict


def _verdict(
    verdict: str = "accept", severity: str = "none", confidence: float = 0.9,
) -> JudgeVerdict:
    return JudgeVerdict(
        sample_id="s1", judge_model="test-model",
        verdict=verdict, severity=severity, confidence=confidence,
    )


class TestMergeJudgeVerdicts:
    def test_all_accept_merges_to_accept(self):
        v, c = merge_judge_verdicts([_verdict(confidence=0.9), _verdict(confidence=0.7)])
        assert v == "accept"
        assert c == 0.7  # min, not average

    def test_any_fatal_severity_rejects_regardless_of_others(self):
        v, c = merge_judge_verdicts([
            _verdict(severity="fatal", confidence=0.1),
            _verdict(confidence=0.99),
        ])
        assert v == "reject"
        assert c == 0.0

    def test_any_reject_verdict_rejects(self):
        v, _ = merge_judge_verdicts([
            _verdict(verdict="reject", confidence=0.5),
            _verdict(verdict="accept", confidence=0.9),
        ])
        assert v == "reject"

    def test_disagreement_without_reject_is_revise(self):
        v, _ = merge_judge_verdicts([
            _verdict(verdict="revise", confidence=0.5),
            _verdict(verdict="accept", confidence=0.9),
        ])
        assert v == "revise"

    def test_confidence_is_minimum_not_average(self):
        _, c = merge_judge_verdicts([
            _verdict(confidence=0.95), _verdict(confidence=0.4), _verdict(confidence=0.8),
        ])
        assert c == 0.4

    def test_single_judge_passthrough(self):
        v, c = merge_judge_verdicts([_verdict(confidence=0.6)])
        assert v == "accept"
        assert c == 0.6


class TestJudgeEnsembleConstruction:
    def test_requires_at_least_two_models(self):
        with pytest.raises(ValueError):
            JudgeEnsemble(["only-one-model"])

    def test_accepts_two_or_more_models(self):
        ensemble = JudgeEnsemble(["model-a", "model-b"])
        assert ensemble.models == ["model-a", "model-b"]
