"""Stage registry.

Provides a singleton ``stage_registry`` that maps stage names to
``Stage`` subclasses, plus a convenience ``register_stage`` shortcut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.registry.base import Registry

if TYPE_CHECKING:
    from forge.core.stage import Stage

stage_registry: Registry[Stage] = Registry()
"""Global registry for pipeline stages."""

register_stage = stage_registry.register
"""Convenience alias — use as ``@register_stage('name')``."""
