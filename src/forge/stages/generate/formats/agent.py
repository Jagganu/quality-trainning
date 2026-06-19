"""Agent format — ReAct observation/thought/action/result."""

from __future__ import annotations
import json
from typing import Any
from forge.core.models import RawGeneration
from forge.stages.generate.formats.base import DatasetFormat


class AgentFormat(DatasetFormat):
    @property
    def name(self) -> str:
        return "agent"

    def get_system_prompt(self) -> str:
        return (
            "You are generating training data for an AI agent that uses tools. "
            "Use the ReAct format: observation, thought, action (with tool + input), result. "
            "Respond ONLY with valid JSON."
        )

    def get_generation_prompt(self, concept: str, context: str, difficulty: str) -> str:
        return (
            f"Generate a {difficulty}-level agent trajectory sample about: {concept}\n\n"
            f"Context:\n{context[:800]}\n\n"
            f'Return JSON: {{"observation": "...", "thought": "...", '
            f'"action": {{"tool": "...", "input": "..."}}, "result": "..."}}'
        )

    def format_sample(self, raw: RawGeneration, concept: str, subtopic: str,
                      difficulty: str, source_doc_ids: list[str]) -> dict[str, Any]:
        try:
            text = raw.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return {"observation": concept, "thought": raw.text, "action": {"tool": "search", "input": concept}, "result": ""}

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        return all(sample.get(k) for k in ("observation", "thought", "action", "result"))

    def schema(self) -> dict[str, Any]:
        return {"type": "object", "required": ["observation", "thought", "action", "result"]}
