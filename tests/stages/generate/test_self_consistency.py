"""Tests for forge.stages.generate.self_consistency — clustering and
winner-selection logic (no real LLM calls; SelfConsistencyGenerator.generate
itself is covered indirectly via GenerateStage integration, these tests
exercise the deterministic clustering/selection helpers directly).
"""

from __future__ import annotations

from forge.core.models import CandidateGeneration, CandidateSet, RawGeneration
from forge.stages.generate.self_consistency import (
    SelfConsistencyGenerator,
    extract_answer_signature,
    normalize_answer,
)


def _candidate(answer_sig: str, programmatic_pass: bool | None = None) -> CandidateGeneration:
    return CandidateGeneration(
        raw=RawGeneration(text="x"),
        content={"answer": answer_sig},
        answer_signature=answer_sig,
        programmatic_pass=programmatic_pass,
    )


# ── normalize_answer ────────────────────────────────────────────────────

class TestNormalizeAnswer:
    def test_lowercases(self):
        assert normalize_answer("ANSWER") == "answer"

    def test_collapses_whitespace(self):
        assert normalize_answer("a   b\n\nc") == "a b c"

    def test_strips_trailing_punctuation(self):
        assert normalize_answer("42.") == "42"
        assert normalize_answer("Is it true?") == "is it true"

    def test_equivalent_phrasing_matches(self):
        assert normalize_answer("The answer is 42.") == normalize_answer("the answer is 42")


# ── extract_answer_signature ────────────────────────────────────────────

class TestExtractAnswerSignature:
    def test_reasoning_format_uses_answer_key(self):
        sig = extract_answer_signature({"answer": "Paris"}, "reasoning")
        assert sig == "paris"

    def test_coding_format_uses_patch_key(self):
        sig = extract_answer_signature({"patch": "def f(): pass"}, "coding")
        assert "def f" in sig

    def test_missing_key_returns_empty(self):
        assert extract_answer_signature({}, "reasoning") == ""

    def test_list_value_joined(self):
        sig = extract_answer_signature({"conversations": ["hi", "there"]}, "chat")
        assert sig == "hi there"


# ── Clustering ───────────────────────────────────────────────────────────

class TestClustering:
    def test_identical_signatures_cluster_together(self):
        candidates = [_candidate("42"), _candidate("42"), _candidate("7")]
        SelfConsistencyGenerator._cluster(candidates)
        assert candidates[0].cluster_id == candidates[1].cluster_id
        assert candidates[0].cluster_id != candidates[2].cluster_id

    def test_largest_cluster_gets_id_zero(self):
        candidates = [_candidate("a"), _candidate("b"), _candidate("b")]
        SelfConsistencyGenerator._cluster(candidates)
        # The "b" pair is the largest group -> cluster_id 0
        b_ids = {c.cluster_id for c in candidates if c.answer_signature == "b"}
        assert b_ids == {0}

    def test_all_unique_each_own_cluster(self):
        candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
        SelfConsistencyGenerator._cluster(candidates)
        assert len({c.cluster_id for c in candidates}) == 3


# ── Winner selection ─────────────────────────────────────────────────────

class TestSelectWinner:
    def test_majority_cluster_wins(self):
        candidates = [_candidate("42"), _candidate("42"), _candidate("7")]
        cset = CandidateSet(candidates=candidates)
        SelfConsistencyGenerator._cluster(candidates)
        SelfConsistencyGenerator._select_winner(cset, candidates)

        winner = next(c for c in candidates if c.candidate_id == cset.selected_id)
        assert winner.answer_signature == "42"
        assert cset.agreement_ratio == 2 / 3
        assert len(cset.rejected_ids) == 2

    def test_no_agreement_leaves_selected_empty(self):
        candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
        cset = CandidateSet(candidates=candidates)
        SelfConsistencyGenerator._cluster(candidates)
        SelfConsistencyGenerator._select_winner(cset, candidates)

        assert cset.selected_id == ""
        assert cset.agreement_ratio == 1 / 3
        assert len(cset.rejected_ids) == 3

    def test_single_candidate_passes_through(self):
        """n=1 (no self-consistency) shouldn't be forced to 'disagree with itself'."""
        candidates = [_candidate("only-one")]
        cset = CandidateSet(candidates=candidates)
        SelfConsistencyGenerator._cluster(candidates)
        SelfConsistencyGenerator._select_winner(cset, candidates)

        assert cset.selected_id == candidates[0].candidate_id

    def test_programmatically_confirmed_member_preferred(self):
        """Within the winning cluster, prefer a candidate with a confirmed
        programmatic pass over one with an undetermined (None) result.
        """
        unconfirmed = _candidate("42", programmatic_pass=None)
        confirmed = _candidate("42", programmatic_pass=True)
        loser = _candidate("7")
        candidates = [unconfirmed, confirmed, loser]
        cset = CandidateSet(candidates=candidates)
        SelfConsistencyGenerator._cluster(candidates)
        SelfConsistencyGenerator._select_winner(cset, candidates)

        assert cset.selected_id == confirmed.candidate_id

    def test_two_way_tie_picks_a_cluster_of_two(self):
        """Two clusters of size 2 each (4 candidates) — either is an
        acceptable winner since both clear the 'agreement' bar.
        """
        candidates = [_candidate("a"), _candidate("a"), _candidate("b"), _candidate("b")]
        cset = CandidateSet(candidates=candidates)
        SelfConsistencyGenerator._cluster(candidates)
        SelfConsistencyGenerator._select_winner(cset, candidates)

        assert cset.selected_id != ""
        assert cset.agreement_ratio == 0.5
