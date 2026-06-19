"""Chat format — ShareGPT conversations."""
from __future__ import annotations
import json
from typing import Any
from forge.core.models import RawGeneration
from forge.stages.generate.formats.base import DatasetFormat

class ChatFormat(DatasetFormat):
    @property
    def name(self) -> str:
        return "chat"

    def get_system_prompt(self) -> str:
        return ("You are generating multi-turn conversation training data in ShareGPT format. "
                "Respond ONLY with valid JSON.")

    def get_generation_prompt(self, concept: str, context: str, difficulty: str) -> str:
        return (f"Generate a {difficulty}-level multi-turn conversation about: {concept}\n\n"
                f"Context:\n{context[:800]}\n\n"
                f'Return JSON: {{"conversations": [{{"from": "human", "value": "..."}}, {{"from": "gpt", "value": "..."}}]}}')

    def format_sample(self, raw: RawGeneration, concept: str, subtopic: str,
                      difficulty: str, source_doc_ids: list[str]) -> dict[str, Any]:
        try:
            text = raw.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return {"conversations": [{"from": "human", "value": concept}, {"from": "gpt", "value": raw.text}]}

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        convs = sample.get("conversations", [])
        return len(convs) >= 2 and all(c.get("from") and c.get("value") for c in convs)

    def schema(self) -> dict[str, Any]:
        return {"type": "object", "required": ["conversations"]}
