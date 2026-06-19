"""Pipeline lifecycle hooks — event system for observability and plugins."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Coroutine

from forge.utils.logging import get_logger

logger = get_logger(__name__)

HookCallback = Callable[..., Coroutine[Any, Any, None]]


class HookEvent(str, Enum):
    """Events emitted during pipeline execution."""
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    BEFORE_STAGE = "before_stage"
    AFTER_STAGE = "after_stage"
    ON_ERROR = "on_error"
    ON_SAMPLE_GENERATED = "on_sample_generated"
    ON_BUDGET_WARNING = "on_budget_warning"


class HookManager:
    """Manages event hooks. One bad hook never crashes the pipeline."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookCallback]] = {e: [] for e in HookEvent}

    def register(self, event: HookEvent, callback: HookCallback) -> None:
        """Register a callback for an event."""
        self._hooks[event].append(callback)
        logger.debug("Hook registered: %s -> %s", event.value, callback.__name__)

    def unregister(self, event: HookEvent, callback: HookCallback) -> None:
        """Remove a callback for an event."""
        try:
            self._hooks[event].remove(callback)
        except ValueError:
            pass

    async def emit(self, event: HookEvent, **kwargs: Any) -> None:
        """Fire all callbacks for *event*. Exceptions are logged, never raised."""
        for cb in self._hooks[event]:
            try:
                await cb(**kwargs)
            except Exception:
                logger.exception("Hook %s raised in %s", event.value, cb.__name__)
