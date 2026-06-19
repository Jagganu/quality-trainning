"""Instruction format — ChatML / OpenAI messages format."""

from __future__ import annotations

import json
from typing import Any

from forge.core.models import RawGeneration
from forge.stages.generate.formats.base import DatasetFormat


class InstructionFormat(DatasetFormat):
    """System/user/assistant message triples (ChatML)."""

    @property
    def name(self) -> str:
        return "instruction"

    def get_system_prompt(self) -> str:
        return (
            "You are a helpful assistant generating training data in ChatML format. "
            "Create realistic system/user/assistant conversation triples. "
            "Respond ONLY with valid JSON."
        )

    def get_generation_prompt(self, concept: str, context: str, difficulty: str) -> str:
        return (
            f"Generate a {difficulty}-level instruction-following training sample about: {concept}\n\n"
            f"Context:\n{context[:800]}\n\n"
            f'Return valid JSON: {{"messages": ['
            f'{{"role": "system", "content": "..."}}, '
            f'{{"role": "user", "content": "..."}}, '
            f'{{"role": "assistant", "content": "..."}}]}}'
        )

    def format_sample(self, raw: RawGeneration, concept: str, subtopic: str,
                      difficulty: str, source_doc_ids: list[str]) -> dict[str, Any]:
        try:
            text = raw.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return {"messages": [
                {"role": "system", "content": f"You are an expert on {subtopic}."},
                {"role": "user", "content": concept},
                {"role": "assistant", "content": raw.text},
            ]}

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        msgs = sample.get("messages", [])
        return len(msgs) >= 2 and all(m.get("role") and m.get("content") for m in msgs)

    def schema(self) -> dict[str, Any]:
        return {"type": "object", "required": ["messages"],
                "properties": {"messages": {"type": "array", "items": {"type": "object"}}}}
