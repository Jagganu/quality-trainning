"""LLM provider registry.

Provides a singleton ``provider_registry`` for registering and
discovering LLM provider implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.registry.base import Registry

if TYPE_CHECKING:
    from forge.providers.llm import LLMProvider

provider_registry: Registry[LLMProvider] = Registry()
"""Global registry for LLM provider back-ends."""

register_provider = provider_registry.register
"""Convenience alias — use as ``@register_provider('openai')``."""
