"""Consensus engine for aggregating verification signals.

The :class:`ConsensusEngine` combines critiques from multiple
:class:`~forge.verification.critics.Critic` instances and scores from
:class:`~forge.verification.scorers.Scorer` instances to produce a single
:class:`ConsensusResult` with a final verdict.
"""

from __future__ import annotations

from forge.utils.logging import get_logger
from forge.verification.models import ConsensusResult, Critique, JudgeVerdict, ScoreResult

logger = get_logger(__name__)


class ConsensusEngine:
    """Aggregate critiques and scores into a final verdict.

    Parameters
    ----------
    min_pass_rate:
        Minimum fraction of critiques that must pass for an ``accept``
        verdict.
    min_score:
        Minimum average overall score for an ``accept`` verdict.
    """

    def __init__(
        self,
        min_pass_rate: float = 0.6,
        min_score: float = 0.7,
    ) -> None:
        self.min_pass_rate = min_pass_rate
        self.min_score = min_score

    def evaluate(
        self,
        sample_id: str,
        critiques: list[Critique],
        scores: list[ScoreResult],
    ) -> ConsensusResult:
        """Produce a :class:`ConsensusResult` from critiques and scores.

        Decision logic
        --------------
        1. If **any** critique has ``severity == "fatal"`` → ``reject``.
        2. If pass-rate ≥ *min_pass_rate* **and** average score ≥
           *min_score* → ``accept``.
        3. Otherwise → ``revise``.

        Confidence is computed as ``avg_score × pass_rate``.
        """
        # --- Pass rate -------------------------------------------------
        if critiques:
            passed_count = sum(1 for c in critiques if c.passed)
            pass_rate = passed_count / len(critiques)
        else:
            pass_rate = 1.0

        # --- Average score ---------------------------------------------
        if scores:
            avg_score = sum(s.overall for s in scores) / len(scores)
        else:
            avg_score = 0.0

        # --- Fatal check -----------------------------------------------
        has_fatal = any(c.severity == "fatal" for c in critiques)

        # --- Verdict ---------------------------------------------------
        if has_fatal:
            verdict = "reject"
            reasoning = "One or more critics reported a fatal issue"
        elif pass_rate >= self.min_pass_rate and avg_score >= self.min_score:
            verdict = "accept"
            reasoning = (
                f"Pass rate {pass_rate:.0%} ≥ {self.min_pass_rate:.0%} and "
                f"avg score {avg_score:.2f} ≥ {self.min_score:.2f}"
            )
        else:
            verdict = "revise"
            parts: list[str] = []
            if pass_rate < self.min_pass_rate:
                parts.append(
                    f"pass rate {pass_rate:.0%} < {self.min_pass_rate:.0%}"
                )
            if avg_score < self.min_score:
                parts.append(
                    f"avg score {avg_score:.2f} < {self.min_score:.2f}"
                )
            reasoning = "Revision needed: " + "; ".join(parts)

        confidence = round(avg_score * pass_rate, 4)

        logger.debug(
            "Consensus for %s: verdict=%s confidence=%.4f",
            sample_id,
            verdict,
            confidence,
        )

        return ConsensusResult(
            sample_id=sample_id,
            critiques=list(critiques),
            scores=list(scores),
            final_verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
        )

    def evaluate_with_judges(
        self,
        sample_id: str,
        critiques: list[Critique],
        judge_verdicts: list[JudgeVerdict],
        agreement_ratio: float = 1.0,
        min_agreement: float = 0.34,
    ) -> ConsensusResult:
        """Merge cheap critics (format/factual — no LLM cost), an already
        cheap-checked self-consistency ``agreement_ratio`` from the
        Generate stage, and judge-ensemble verdicts into one final result.

        Decision logic
        --------------
        1. Any fatal critique, or the judge ensemble's merged verdict is
           ``reject`` → ``reject``.
        2. Judge ensemble says ``accept`` AND agreement_ratio ≥
           *min_agreement* AND critique pass_rate ≥ ``self.min_pass_rate``
           → ``accept``. Pass rate is a ratio, not "every critique must
           pass" — consistent with the original evaluate() semantics, so
           a single minor critique (e.g. a heuristic factual-overlap flag)
           doesn't unconditionally veto an otherwise-good sample.
        3. Otherwise → ``revise``. Low self-consistency alone (candidates
           disagreed) is a soft signal, not an automatic reject — a strong
           judge ensemble can still override it, but it pulls confidence
           down via the same min-across-signals rule judges use internally.
        """
        has_fatal_critique = any(c.severity == "fatal" for c in critiques)
        pass_rate = (
            sum(1 for c in critiques if c.passed) / len(critiques) if critiques else 1.0
        )
        critiques_passed = pass_rate >= self.min_pass_rate

        if not judge_verdicts:
            judge_verdict, judge_confidence = "revise", 0.0
        else:
            from forge.verification.judge_ensemble import merge_judge_verdicts
            judge_verdict, judge_confidence = merge_judge_verdicts(judge_verdicts)

        confidence = round(min(judge_confidence, agreement_ratio), 4)

        if has_fatal_critique or judge_verdict == "reject":
            verdict = "reject"
            reasoning = "Fatal critique or judge-ensemble reject"
        elif (
            judge_verdict == "accept"
            and agreement_ratio >= min_agreement
            and critiques_passed
        ):
            verdict = "accept"
            reasoning = (
                f"Judges accepted, agreement {agreement_ratio:.0%} ≥ "
                f"{min_agreement:.0%}, critics passed"
            )
        else:
            verdict = "revise"
            reasoning = (
                f"judge_verdict={judge_verdict}, agreement={agreement_ratio:.0%}, "
                f"critics_passed={critiques_passed}"
            )

        return ConsensusResult(
            sample_id=sample_id,
            critiques=list(critiques),
            final_verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            judge_verdicts=list(judge_verdicts),
            agreement_ratio=agreement_ratio,
        )
