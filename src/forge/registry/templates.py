"""Dataset template registry.

Provides a singleton ``template_registry`` for registering and
discovering dataset template implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.registry.base import Registry

if TYPE_CHECKING:
    from forge.templates.base import DatasetTemplate

template_registry: Registry[DatasetTemplate] = Registry()
"""Global registry for dataset templates."""

register_template = template_registry.register
"""Convenience alias — use as ``@register_template('qa')``."""
