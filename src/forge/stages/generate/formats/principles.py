"""Principles format — First-principles reasoning with explicit WHY rationale."""

from __future__ import annotations

import json
from typing import Any

from forge.core.models import RawGeneration
from forge.stages.generate.formats.base import DatasetFormat


class PrinciplesFormat(DatasetFormat):
    """Scenario → First Principles → Why Rationale → Principle-Adhering Solution."""

    @property
    def name(self) -> str:
        return "principles"

    def get_system_prompt(self) -> str:
        return (
            "You are an expert first-principles instructor. For every scenario, you must analyze "
            "it from the ground up. Identify the core scientific, mathematical, or engineering "
            "principles that govern the situation, explain exactly WHY they apply, and derive "
            "a solution that strictly adheres to those principles. Do not skip logical steps."
        )

    def get_generation_prompt(self, concept: str, context: str, difficulty: str) -> str:
        return (
            f"Generate a {difficulty}-level first-principles training sample about: {concept}\n\n"
            f"Context from source material:\n{context[:1000]}\n\n"
            f"Your output must be a valid JSON with this exact structure:\n"
            f'{{\n'
            f'  "scenario": "A clear, realistic question, coding issue, or scenario about {concept}.",\n'
            f'  "core_principles": ["Principle 1 (e.g., Single Responsibility)", "Principle 2 (e.g., Least Privilege)"],\n'
            f'  "why_rationale": "A detailed, rigorous explanation of WHY these principles apply here, the logical deduction from first-principles, and why other intuitive but incorrect paths fail.",\n'
            f'  "solution": "The definitive, optimal action, code, or answer that perfectly implements the principles.",\n'
            f'  "boundary_conditions": "Explain the limits of these principles in this scenario—when do they not apply, or how do they trade off with other principles?",\n'
            f'  "metadata": {{"difficulty": "{difficulty}", "subtopic": "{concept}"}}\n'
            f'}}'
        )

    def format_sample(self, raw: RawGeneration, concept: str, subtopic: str,
                      difficulty: str, source_doc_ids: list[str]) -> dict[str, Any]:
        try:
            text = raw.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            data = {
                "scenario": concept,
                "core_principles": [],
                "why_rationale": raw.text,
                "solution": "",
                "boundary_conditions": "",
                "metadata": {}
            }

        data.setdefault("metadata", {})
        data["metadata"]["difficulty"] = difficulty
        data["metadata"]["subtopic"] = subtopic
        return data

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        required = ("scenario", "core_principles", "why_rationale", "solution", "boundary_conditions")
        return all(sample.get(k) for k in required)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["scenario", "core_principles", "why_rationale", "solution", "boundary_conditions", "metadata"],
            "properties": {
                "scenario": {"type": "string"},
                "core_principles": {"type": "array", "items": {"type": "string"}},
                "why_rationale": {"type": "string"},
                "solution": {"type": "string"},
                "boundary_conditions": {"type": "string"},
                "metadata": {"type": "object"},
            },
        }