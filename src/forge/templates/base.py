"""Template ABC — domain-specific dataset generation blueprints."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Template(ABC):
    """Base class for all dataset generation templates.

    A template encapsulates domain-specific knowledge: what to search for,
    how to guide generation, and expected scale.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'cybersecurity', 'coding')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the template."""

    @property
    @abstractmethod
    def default_format(self) -> str:
        """Default dataset format for this template (e.g. 'reasoning')."""

    @abstractmethod
    def seed_topics(self) -> list[str]:
        """Initial topics / search queries for the Collect stage."""

    @abstractmethod
    def generation_guidelines(self) -> str:
        """Extra instructions included in generation prompts."""

    def estimated_page_count(self) -> int:
        """Estimated number of web pages to collect."""
        return 30

    def estimated_sample_count(self) -> int:
        """Estimated number of training samples to generate."""
        return 100
