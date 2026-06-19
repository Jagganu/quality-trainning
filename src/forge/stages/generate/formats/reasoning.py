"""Reasoning format — v3 schema with analysis field."""

from __future__ import annotations

import json
from typing import Any

from forge.core.models import RawGeneration
from forge.stages.generate.formats.base import DatasetFormat


class ReasoningFormat(DatasetFormat):
    """Question → step-by-step analysis → answer."""

    @property
    def name(self) -> str:
        return "reasoning"

    def get_system_prompt(self) -> str:
        return (
            "You are an expert reasoning assistant. For every question, provide:\n"
            "1. A clear, specific question\n"
            "2. Detailed step-by-step analysis showing your reasoning\n"
            "3. A concise, definitive answer\n\n"
            "Respond ONLY with valid JSON matching the required schema."
        )

    def get_generation_prompt(self, concept: str, context: str, difficulty: str) -> str:
        return (
            f"Generate a {difficulty}-level reasoning training sample about: {concept}\n\n"
            f"Context from source material:\n{context[:1000]}\n\n"
            f"Return valid JSON:\n"
            f'{{"question": "...", "analysis": "step-by-step reasoning...", '
            f'"answer": "...", "metadata": {{"difficulty": "{difficulty}", '
            f'"subtopic": "{concept}", "approach": "step_by_step", "confidence": 0.9}}}}'
        )

    def format_sample(self, raw: RawGeneration, concept: str, subtopic: str,
                      difficulty: str, source_doc_ids: list[str]) -> dict[str, Any]:
        try:
            text = raw.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            data = {"question": concept, "analysis": raw.text, "answer": "", "metadata": {}}

        data.setdefault("metadata", {})
        data["metadata"]["difficulty"] = difficulty
        data["metadata"]["subtopic"] = subtopic
        return data

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        required = ("question", "analysis", "answer")
        return all(sample.get(k) for k in required)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["question", "analysis", "answer", "metadata"],
            "properties": {
                "question": {"type": "string"},
                "analysis": {"type": "string"},
                "answer": {"type": "string"},
                "metadata": {"type": "object"},
            },
        }
