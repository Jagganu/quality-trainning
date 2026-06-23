"""Refine stage — re-generates samples that received a 'revise' verdict.

Refined samples are re-verified through the same critics/consensus used
in VerifyStage before being marked accepted — a refine pass is a proposal,
not a guarantee, and the previous version of this stage accepted every
refinement unconditionally without checking whether it actually fixed
anything. Each sample gets up to ``settings.refine.max_retries`` refine →
re-verify round trips; if it's still flagged 'revise' after that, it's
left in its last state for the pipeline's quality gates to deal with.
"""

from __future__ import annotations

import json

from forge.core.context import PipelineContext
from forge.core.models import SampleVerdict
from forge.core.stage import Stage
from forge.providers.llm import LLMProvider
from forge.utils.logging import get_logger
from forge.verification.consensus import ConsensusEngine
from forge.verification.critics import FactualCritic, FormatCritic, LLMCritic
from forge.verification.scorers import LLMScorer

logger = get_logger(__name__)

_VERDICT_MAP = {
    "accept": SampleVerdict.ACCEPT,
    "reject": SampleVerdict.REJECT,
    "revise": SampleVerdict.REVISE,
}


class RefineStage(Stage):
    """Stage 5: Refine samples flagged for revision, then re-verify them."""

    @property
    def name(self) -> str:
        return "refine"

    async def run(self, context: PipelineContext) -> PipelineContext:
        llm = LLMProvider(context.settings, context.budget)
        verify_settings = context.settings.verify
        max_retries = max(1, context.settings.refine.max_retries)

        to_revise = [
            s for s in context.samples
            if s.verification and s.verification.verdict == SampleVerdict.REVISE
        ]

        if not to_revise:
            logger.info("No samples require refinement")
            return context

        # Reuse the same cheap critics + (optional) LLM critic/scorer the
        # Verify stage uses, so "refined" actually means "passes the same
        # bar", not just "an LLM said it tried".
        critics = [FormatCritic(), FactualCritic()]
        critic_model = verify_settings.critic_model or context.settings.default_model
        scorer = None
        if critic_model:
            critics.append(LLMCritic(model=critic_model))
            scorer = LLMScorer(model=verify_settings.scorer_model or context.settings.default_model)
        consensus = ConsensusEngine(
            min_pass_rate=verify_settings.min_pass_rate,
            min_score=verify_settings.min_score,
        )
        source_docs = context.documents

        logger.info("Refining %d samples (max %d attempts each)…", len(to_revise), max_retries)
        accepted_count = 0
        still_revising = 0
        rejected_count = 0

        for sample in to_revise:
            for attempt in range(1, max_retries + 1):
                issues = sample.verification.issues if sample.verification else []
                original_content = json.dumps(sample.content, indent=2, default=str)

                prompt = (
                    "You are improving an AI training sample. The original sample "
                    "had quality issues.\n\n"
                    f"ORIGINAL SAMPLE:\n{original_content}\n\n"
                    "ISSUES FOUND:\n" + "\n".join(f"- {i}" for i in issues) + "\n\n"
                    "Fix ALL listed issues and return the corrected sample as valid JSON. "
                    "Keep the same structure and format. Only fix the problems — don't change "
                    "content that is already correct."
                )

                try:
                    raw = await llm.complete(
                        prompt=prompt,
                        system="You are a data-quality expert. Return only valid JSON.",
                        stage="refine",
                    )
                    text = raw.text.strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                    sample.content = json.loads(text)
                    sample.lineage.stage_versions["refine"] = f"1.0-attempt{attempt}"
                except Exception as exc:
                    logger.warning(
                        "Refine attempt %d failed for %s: %s (keeping previous content)",
                        attempt, sample.lineage.sample_id, exc,
                    )
                    # Generation failed — nothing changed, so re-verifying
                    # would just repeat the same revise verdict. Try again
                    # if attempts remain, otherwise fall through and exit.
                    if attempt < max_retries:
                        continue
                    break

                # Re-verify the refined content against the same bar
                # Verify uses — this is the part the old implementation
                # skipped entirely.
                sid = sample.lineage.sample_id
                critiques = []
                for critic in critics:
                    try:
                        critiques.append(await critic.critique(sample, source_docs))
                    except Exception as exc:
                        logger.warning(
                            "Critic %s failed for %s: %s", type(critic).__name__, sid, exc,
                        )

                scores = []
                if scorer:
                    try:
                        scores.append(await scorer.score(sample))
                    except Exception as exc:
                        logger.warning("Scorer failed for %s: %s", sid, exc)

                result = consensus.evaluate(sid, critiques, scores)
                verdict = _VERDICT_MAP.get(result.final_verdict, SampleVerdict.REVISE)

                sample.verification.verdict = verdict
                sample.verification.score = result.confidence
                sample.verification.issues = [issue for c in critiques for issue in c.issues]
                sample.verification.reasoning = f"Refine attempt {attempt}: {result.reasoning}"
                sample.quality_score = result.confidence

                if verdict != SampleVerdict.REVISE:
                    break  # accepted or rejected — stop retrying either way

            if sample.verification.verdict == SampleVerdict.ACCEPT:
                accepted_count += 1
            elif sample.verification.verdict == SampleVerdict.REJECT:
                rejected_count += 1
            else:
                still_revising += 1

        context.metrics.increment("samples_refined", accepted_count)
        context.metrics.increment("samples_refine_rejected", rejected_count)
        context.metrics.increment("samples_refine_still_revise", still_revising)
        logger.info(
            "Refine complete: %d accepted, %d rejected, %d still need revision (of %d)",
            accepted_count, rejected_count, still_revising, len(to_revise),
        )

        return context
