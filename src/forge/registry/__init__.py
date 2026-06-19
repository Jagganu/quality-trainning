"""Plugin registry subsystem.

Exports
-------
Registry
    Generic, type-safe plugin registry.
stage_registry
    Registry instance for pipeline stages.
provider_registry
    Registry instance for LLM providers.
template_registry
    Registry instance for dataset templates.
"""

from __future__ import annotations

from forge.registry.base import Registry
from forge.registry.providers import provider_registry, register_provider
from forge.registry.stages import register_stage, stage_registry
from forge.registry.templates import register_template, template_registry

__all__ = [
    "Registry",
    "provider_registry",
    "register_provider",
    "register_stage",
    "register_template",
    "stage_registry",
    "template_registry",
]
