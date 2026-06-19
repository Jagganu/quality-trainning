"""Coding format — issue/investigation/patch/verification."""
from __future__ import annotations
import json
from typing import Any
from forge.core.models import RawGeneration
from forge.stages.generate.formats.base import DatasetFormat

class CodingFormat(DatasetFormat):
    @property
    def name(self) -> str:
        return "coding"

    def get_system_prompt(self) -> str:
        return ("You are a senior software engineer generating training data for code debugging and review. "
                "Respond ONLY with valid JSON.")

    def get_generation_prompt(self, concept: str, context: str, difficulty: str) -> str:
        return (f"Generate a {difficulty}-level coding training sample about: {concept}\n\n"
                f"Context:\n{context[:800]}\n\n"
                f'Return JSON: {{"issue": "...", "investigation": "...", "patch": "...", "verification": "..."}}')

    def format_sample(self, raw: RawGeneration, concept: str, subtopic: str,
                      difficulty: str, source_doc_ids: list[str]) -> dict[str, Any]:
        try:
            text = raw.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return {"issue": concept, "investigation": raw.text, "patch": "", "verification": ""}

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        return all(sample.get(k) for k in ("issue", "investigation", "patch"))

    def schema(self) -> dict[str, Any]:
        return {"type": "object", "required": ["issue", "investigation", "patch", "verification"]}
