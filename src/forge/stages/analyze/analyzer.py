"""Analyze stage — LLM-powered corpus analysis to guide generation."""

from __future__ import annotations

import json

from forge.core.context import PipelineContext
from forge.core.models import AnalysisResult, GenerationPlan, SubtopicPlan
from forge.core.stage import Stage
from forge.providers.llm import LLMProvider
from forge.utils.logging import get_logger

logger = get_logger(__name__)

_ANALYZE_PROMPT = """\
Analyze the following corpus excerpts about "{topic}" and produce a structured generation plan.

CORPUS EXCERPTS:
{corpus}

Return ONLY valid JSON with this structure:
{{
  "subtopics": ["list of 5-10 key subtopics"],
  "concepts": ["list of 15-30 specific concepts to generate training samples about"],
  "sample_types": ["explanation", "problem_solving", "analysis", "comparison"],
  "generation_plan": [
    {{
      "name": "subtopic_name",
      "concepts": ["concept1", "concept2"],
      "sample_count": 10,
      "difficulties": {{"beginner": 3, "intermediate": 4, "advanced": 3}}
    }}
  ]
}}
"""


class AnalyzeStage(Stage):
    """Stage 3: Analyze the corpus and build a generation plan."""

    @property
    def name(self) -> str:
        return "analyze"

    async def run(self, context: PipelineContext) -> PipelineContext:
        llm = LLMProvider(context.settings, context.budget)

        # Build corpus summary from cleaned chunks
        all_text = []
        for doc in context.cleaned_documents:
            for chunk in doc.chunks[:2]:  # First 2 chunks per doc
                all_text.append(chunk[:500])
        corpus_summary = "\n---\n".join(all_text[:20])  # Cap at 20 excerpts

        if not corpus_summary:
            corpus_summary = f"Topic: {context.topic}. No corpus collected yet."

        prompt = _ANALYZE_PROMPT.format(topic=context.topic, corpus=corpus_summary)
        raw = await llm.complete(
            prompt=prompt,
            system="You are a dataset architect. Analyze content and produce structured generation plans. Return only valid JSON.",
            stage="analyze",
        )

        # Parse response
        try:
            text = raw.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse analysis JSON, using defaults")
            data = {
                "subtopics": [context.topic],
                "concepts": [context.topic],
                "sample_types": ["explanation"],
                "generation_plan": [{"name": context.topic, "concepts": [context.topic], "sample_count": 10}],
            }

        analysis = AnalysisResult(
            topic=context.topic,
            subtopics=data.get("subtopics", []),
            concepts=data.get("concepts", []),
            sample_types=data.get("sample_types", []),
        )

        # Build generation plan
        plans = []
        for item in data.get("generation_plan", []):
            plans.append(SubtopicPlan(
                name=item.get("name", ""),
                concepts=item.get("concepts", []),
                sample_count=item.get("sample_count", 10),
                difficulties=item.get("difficulties", {"beginner": 3, "intermediate": 4, "advanced": 3}),
            ))
        analysis.generation_plan = plans

        total = sum(p.sample_count for p in plans)
        context.analysis = analysis
        context.generation_plan = GenerationPlan(
            topic=context.topic,
            subtopics=plans,
            total_samples=total,
            format=context.settings.generate.default_format,
        )

        context.metrics.increment("subtopics_found", len(analysis.subtopics))
        context.metrics.increment("concepts_found", len(analysis.concepts))
        logger.info("Analysis: %d subtopics, %d concepts, %d planned samples",
                     len(analysis.subtopics), len(analysis.concepts), total)
        return context
