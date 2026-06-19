"""Refine stage — re-generates samples that received a 'revise' verdict."""

from __future__ import annotations

import json

from forge.core.context import PipelineContext
from forge.core.models import SampleVerdict
from forge.core.stage import Stage
from forge.providers.llm import LLMProvider
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class RefineStage(Stage):
    """Stage 5: Refine samples flagged for revision."""

    @property
    def name(self) -> str:
        return "refine"

    async def run(self, context: PipelineContext) -> PipelineContext:
        llm = LLMProvider(context.settings, context.budget)

        to_revise = [
            s for s in context.samples
            if s.verification and s.verification.verdict == SampleVerdict.REVISE
        ]

        if not to_revise:
            logger.info("No samples require refinement")
            return context

        logger.info("Refining %d samples…", len(to_revise))
        refined_count = 0

        for sample in to_revise:
            issues = sample.verification.issues if sample.verification else []
            original_content = json.dumps(sample.content, indent=2, default=str)

            prompt = (
                "You are improving an AI training sample. The original sample had quality issues.\n\n"
                f"ORIGINAL SAMPLE:\n{original_content}\n\n"
                f"ISSUES FOUND:\n" + "\n".join(f"- {i}" for i in issues) + "\n\n"
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
                new_content = json.loads(text)
                sample.content = new_content
                sample.lineage.stage_versions["refine"] = "1.0"
                # Reset verdict to revise-completed (will need re-verification in production)
                if sample.verification:
                    sample.verification.verdict = SampleVerdict.ACCEPT
                    sample.verification.reasoning = "Refined based on critique feedback"
                refined_count += 1
            except Exception as exc:
                logger.warning(
                    "Failed to refine sample %s: %s (keeping original)",
                    sample.lineage.sample_id, exc,
                )

        context.metrics.increment("samples_refined", refined_count)
        logger.info("Refined %d/%d samples", refined_count, len(to_revise))

        return context
