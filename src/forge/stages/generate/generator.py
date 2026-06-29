"""Generate stage — produces training samples from cleaned documents."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone

from forge.core.context import PipelineContext
from forge.core.models import Sample, SampleLineage
from forge.core.stage import Stage
from forge.providers.llm import LLMProvider
from forge.stages.generate.diversity import DiversityAnalyzer
from forge.stages.generate.formats import FORMAT_REGISTRY
from forge.stages.generate.self_consistency import SelfConsistencyGenerator
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class GenerateStage(Stage):
    """Stage 3: Generate training samples from cleaned documents."""

    @property
    def name(self) -> str:
        return "generate"

    async def run(self, context: PipelineContext) -> PipelineContext:
        settings = context.settings.generate
        llm = LLMProvider(context.settings, context.budget)

        # Resolve output format
        fmt_cls = FORMAT_REGISTRY.get(settings.default_format)
        if fmt_cls is None:
            logger.warning(
                "Unknown format '%s', falling back to 'reasoning'",
                settings.default_format,
            )
            fmt_cls = FORMAT_REGISTRY["reasoning"]
        fmt = fmt_cls()

        # Build context chunks from cleaned documents
        chunks: list[str] = []
        doc_ids: list[str] = []
        for doc in context.cleaned_documents:
            for chunk in doc.chunks:
                chunks.append(chunk)
                doc_ids.append(doc.doc_id)

        if not chunks:
            logger.warning("No cleaned document chunks available — using topic as context")
            chunks = [context.topic]
            doc_ids = ["topic"]

        # Determine generation plan
        plan = context.analysis
        if plan and plan.generation_plan:
            subtopic_plans = plan.generation_plan
        else:
            # Fallback: generate from topic directly
            from forge.core.models import SubtopicPlan
            subtopic_plans = [
                SubtopicPlan(
                    name=context.topic,
                    concepts=[context.topic],
                    sample_count=settings.max_samples,
                )
            ]

        # Generate samples
        all_samples: list[Sample] = []
        total_target = min(
            settings.max_samples,
            sum(sp.sample_count for sp in subtopic_plans),
        )
        remaining = total_target

        for sp in subtopic_plans:
            if remaining <= 0:
                break

            for concept in sp.concepts or [sp.name]:
                if remaining <= 0:
                    break

                for difficulty, count in sp.difficulties.items():
                    batch_size = min(count, remaining, settings.batch_size)
                    if batch_size <= 0:
                        break

                    # Pick random context chunks
                    ctx_chunks = random.sample(chunks, min(3, len(chunks)))
                    context_text = "\n\n".join(ctx_chunks)
                    source_ids = random.sample(doc_ids, min(3, len(doc_ids)))
                    prompt_text = fmt.get_generation_prompt(concept, context_text, difficulty)

                    if settings.self_consistency_n > 1:
                        new_samples = await self._generate_self_consistent(
                            llm, fmt, settings, batch_size, prompt_text,
                            concept, sp, difficulty, source_ids, context,
                        )
                    else:
                        new_samples = await self._generate_single_shot(
                            llm, fmt, settings, batch_size, prompt_text,
                            concept, sp, difficulty, source_ids, context,
                        )

                    all_samples.extend(new_samples)
                    remaining -= len(new_samples)

        # Diversity analysis
        analyzer = DiversityAnalyzer()
        context.diversity_score = analyzer.compute_score(all_samples)

        context.samples = all_samples
        context.metrics.increment("samples_generated", len(all_samples))
        logger.info(
            "Generated %d samples (target: %d, diversity: %.2f)",
            len(all_samples), total_target,
            context.diversity_score.overall if context.diversity_score else 0,
        )

        # Persist samples
        for sample in all_samples:
            await context.storage.save_document(
                "samples", sample.lineage.sample_id, sample.model_dump(),
            )

        return context

    async def _generate_single_shot(
        self, llm, fmt, settings, batch_size, prompt_text,
        concept, sp, difficulty, source_ids, context,
    ) -> list[Sample]:
        """Original path: 1 candidate per sample, no clustering."""
        logger.info(
            "Generating %d %s samples for '%s' (%s)…",
            batch_size, difficulty, concept, fmt.name,
        )
        raws = await llm.complete_batch(
            [prompt_text] * batch_size,
            system=fmt.get_system_prompt(),
            concurrency=settings.max_concurrent,
            stage="generate",
        )

        samples: list[Sample] = []
        for raw in raws:
            content = fmt.format_sample(raw, concept, sp.name, difficulty, source_ids)
            if not fmt.validate_sample(content):
                context.metrics.increment("samples_invalid", 1)
                continue
            samples.append(self._build_sample(content, raw.model, source_ids, fmt, context))
        return samples

    async def _generate_self_consistent(
        self, llm, fmt, settings, batch_size, prompt_text,
        concept, sp, difficulty, source_ids, context,
    ) -> list[Sample]:
        """Self-consistency path: N candidates per sample, clustered and
        selected before the sample even reaches Verify. Samples with no
        cluster agreement are dropped here (counted as samples_invalid)
        rather than passed downstream to waste a judge-ensemble call on
        something already known to be inconsistent.
        """
        sc = SelfConsistencyGenerator(
            llm, n=settings.self_consistency_n, models=settings.self_consistency_models,
        )
        logger.info(
            "Self-consistency: generating %d x %d candidates for '%s' (%s)…",
            batch_size, settings.self_consistency_n, concept, fmt.name,
        )

        def make_format_fn():
            return lambda raw: fmt.format_sample(raw, concept, sp.name, difficulty, source_ids)

        semaphore = asyncio.Semaphore(settings.max_concurrent)

        async def _one() -> Sample | None:
            async with semaphore:
                cset = await sc.generate(
                    prompt=prompt_text,
                    system=fmt.get_system_prompt(),
                    fmt_name=fmt.name,
                    format_sample_fn=make_format_fn(),
                )
            if not cset.selected_id:
                context.metrics.increment("samples_invalid", 1)
                return None

            winner = next(c for c in cset.candidates if c.candidate_id == cset.selected_id)
            content = winner.content
            if not fmt.validate_sample(content):
                context.metrics.increment("samples_invalid", 1)
                return None

            content.setdefault("metadata", {})
            content["metadata"]["self_consistency_agreement"] = cset.agreement_ratio
            return self._build_sample(content, winner.raw.model, source_ids, fmt, context)

        results = await asyncio.gather(*(_one() for _ in range(batch_size)))
        return [s for s in results if s is not None]

    @staticmethod
    def _build_sample(content, model, source_ids, fmt, context) -> Sample:
        return Sample(
            lineage=SampleLineage(
                source_documents=list(source_ids),
                generation_model=model,
                generation_timestamp=datetime.now(tz=timezone.utc),
                template=context.template_name,
                pipeline_run_id=context.run_id,
                format=fmt.name,
                stage_versions={"generate": "1.0"},
            ),
            content=content,
        )
