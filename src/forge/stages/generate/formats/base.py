"""Dataset format ABC — all output formats implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from forge.core.models import RawGeneration


class DatasetFormat(ABC):
    """Interface for dataset output formats (reasoning, instruction, agent, …)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def format_sample(self, raw: RawGeneration, concept: str, subtopic: str,
                      difficulty: str, source_doc_ids: list[str]) -> dict[str, Any]:
        """Transform raw LLM output into format-specific dict."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str: ...

    @abstractmethod
    def get_generation_prompt(self, concept: str, context: str, difficulty: str) -> str: ...

    @abstractmethod
    def validate_sample(self, sample: dict[str, Any]) -> bool: ...

    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """JSON schema for this format."""
        ...
