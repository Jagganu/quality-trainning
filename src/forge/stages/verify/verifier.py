"""Verify stage — runs critics, scorers, and consensus on every sample."""

from __future__ import annotations

from forge.core.context import PipelineContext
from forge.core.models import SampleVerdict, VerificationResult
from forge.core.stage import Stage
from forge.verification.consensus import ConsensusEngine
from forge.verification.critics import FactualCritic, FormatCritic, LLMCritic
from forge.verification.scorers import LLMScorer
from forge.utils.logging import get_logger

logger = get_logger(__name__)


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

        # Build critics (format + factual are always used; LLM is optional)
        critics = [FormatCritic(), FactualCritic()]
        use_llm = bool(settings.critic_model)
        if use_llm:
            critics.append(LLMCritic(model=settings.critic_model))

        # Build scorer
        scorer = LLMScorer(model=settings.scorer_model) if use_llm else None

        # Consensus engine
        consensus = ConsensusEngine(
            min_pass_rate=settings.min_pass_rate,
            min_score=settings.min_score,
        )

        # Gather source docs for factual grounding checks
        source_docs = context.documents

        accepted = 0
        rejected = 0
        revised = 0

        for sample in context.samples:
            sid = sample.lineage.sample_id

            # Run all critics
            critiques = []
            for critic in critics:
                try:
                    critique = await critic.critique(sample, source_docs)
                    critiques.append(critique)
                except Exception as exc:
                    logger.warning("Critic %s failed for %s: %s", type(critic).__name__, sid, exc)

            # Run scorer
            scores = []
            if scorer:
                try:
                    score = await scorer.score(sample)
                    scores.append(score)
                except Exception as exc:
                    logger.warning("Scorer failed for %s: %s", sid, exc)

            # Consensus
            result = consensus.evaluate(sid, critiques, scores)

            # Map to VerificationResult
            verdict_map = {
                "accept": SampleVerdict.ACCEPT,
                "reject": SampleVerdict.REJECT,
                "revise": SampleVerdict.REVISE,
            }
            verdict = verdict_map.get(result.final_verdict, SampleVerdict.REVISE)

            all_issues = []
            for c in critiques:
                all_issues.extend(c.issues)

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
