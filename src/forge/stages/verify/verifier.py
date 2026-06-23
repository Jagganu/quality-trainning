"""Verify stage — runs critics, scorers, and consensus on every sample."""

from __future__ import annotations

from forge.core.context import PipelineContext
from forge.core.models import SampleVerdict, VerificationResult
from forge.core.stage import Stage
from forge.utils.logging import get_logger
from forge.verification.consensus import ConsensusEngine
from forge.verification.critics import FactualCritic, FormatCritic, LLMCritic
from forge.verification.judge_ensemble import JudgeEnsemble
from forge.verification.scorers import LLMScorer

logger = get_logger(__name__)

_VERDICT_MAP = {
    "accept": SampleVerdict.ACCEPT,
    "reject": SampleVerdict.REJECT,
    "revise": SampleVerdict.REVISE,
}


class VerifyStage(Stage):
    """Stage 4: Verify quality of generated samples."""

    @property
    def name(self) -> str:
        return "verify"

    async def run(self, context: PipelineContext) -> PipelineContext:
        settings = context.settings.verify
        if not settings.enabled:
            logger.info("Verification disabled — skipping")
            return context

        if not context.samples:
            logger.warning("No samples to verify")
            return context

        # Cheap, non-LLM critics always run regardless of which path below
        # is used — they're free and catch structural problems.
        base_critics = [FormatCritic(), FactualCritic()]
        source_docs = context.documents

        if len(settings.judge_models) >= 2:
            accepted, rejected, revised = await self._run_judge_ensemble(
                context, base_critics, source_docs, settings,
            )
        else:
            accepted, rejected, revised = await self._run_single_critic(
                context, base_critics, source_docs, settings,
            )

        total = len(context.samples)
        pass_rate = accepted / total if total else 0
        context.metrics.increment("samples_accepted", accepted)
        context.metrics.increment("samples_rejected", rejected)
        context.metrics.increment("samples_revised", revised)

        logger.info(
            "Verification complete: %d accepted, %d rejected, %d to revise (%.0f%% pass rate)",
            accepted, rejected, revised, pass_rate * 100,
        )

        return context

    async def _run_single_critic(self, context, base_critics, source_docs, settings):
        """Original path: single critic model + single scorer model."""
        critics = list(base_critics)
        critic_model = settings.critic_model or context.settings.default_model
        scorer_model = settings.scorer_model or context.settings.default_model
        use_llm = bool(critic_model)
        if use_llm:
            critics.append(LLMCritic(model=critic_model))

        scorer = LLMScorer(model=scorer_model) if use_llm else None
        consensus = ConsensusEngine(
            min_pass_rate=settings.min_pass_rate,
            min_score=settings.min_score,
        )

        accepted = rejected = revised = 0
        for sample in context.samples:
            sid = sample.lineage.sample_id

            critiques = []
            for critic in critics:
                try:
                    critiques.append(await critic.critique(sample, source_docs))
                except Exception as exc:
                    logger.warning("Critic %s failed for %s: %s", type(critic).__name__, sid, exc)

            scores = []
            if scorer:
                try:
                    scores.append(await scorer.score(sample))
                except Exception as exc:
                    logger.warning("Scorer failed for %s: %s", sid, exc)

            result = consensus.evaluate(sid, critiques, scores)
            verdict = _VERDICT_MAP.get(result.final_verdict, SampleVerdict.REVISE)
            all_issues = [issue for c in critiques for issue in c.issues]

            sample.verification = VerificationResult(
                sample_id=sid,
                verdict=verdict,
                score=result.confidence,
                issues=all_issues,
                critic_model=settings.critic_model,
                scorer_model=settings.scorer_model,
                reasoning=result.reasoning,
            )
            sample.quality_score = result.confidence

            if verdict == SampleVerdict.ACCEPT:
                accepted += 1
            elif verdict == SampleVerdict.REJECT:
                rejected += 1
            else:
                revised += 1

        return accepted, rejected, revised

    async def _run_judge_ensemble(self, context, base_critics, source_docs, settings):
        """Multi-model judge ensemble path — requires cross-model agreement
        to accept instead of a single critic's opinion.

        Self-consistency's agreement_ratio is read off each sample's
        ``content["metadata"]["self_consistency_agreement"]`` when the
        Generate stage populated it; defaults to 1.0 (neutral) for samples
        generated via the original single-candidate path so this still
        works standalone.
        """
        ensemble = JudgeEnsemble(settings.judge_models)
        consensus = ConsensusEngine(
            min_pass_rate=settings.min_pass_rate,
            min_score=settings.min_score,
        )

        accepted = rejected = revised = 0
        for sample in context.samples:
            sid = sample.lineage.sample_id

            critiques = []
            for critic in base_critics:
                try:
                    critiques.append(await critic.critique(sample, source_docs))
                except Exception as exc:
                    logger.warning("Critic %s failed for %s: %s", type(critic).__name__, sid, exc)

            judge_verdicts, _, _ = await ensemble.evaluate(sample)
            agreement_ratio = sample.content.get("metadata", {}).get(
                "self_consistency_agreement", 1.0
            )

            result = consensus.evaluate_with_judges(
                sid, critiques, judge_verdicts, agreement_ratio=agreement_ratio,
            )
            verdict = _VERDICT_MAP.get(result.final_verdict, SampleVerdict.REVISE)
            all_issues = [issue for c in critiques for issue in c.issues]
            all_issues += [
                f"[{jv.judge_model}] {jv.reasoning}"
                for jv in judge_verdicts if jv.verdict != "accept"
            ]

            sample.verification = VerificationResult(
                sample_id=sid,
                verdict=verdict,
                score=result.confidence,
                issues=all_issues,
                critic_model=",".join(settings.judge_models),
                scorer_model="",
                reasoning=result.reasoning,
            )
            sample.quality_score = result.confidence

            if verdict == SampleVerdict.ACCEPT:
                accepted += 1
            elif verdict == SampleVerdict.REJECT:
                rejected += 1
            else:
                revised += 1

        return accepted, rejected, revised
