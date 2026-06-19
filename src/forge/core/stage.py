"""Stage abstract base class — every pipeline stage implements this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.core.context import PipelineContext


class Stage(ABC):
    """Base class for all pipeline stages.

    Subclasses must set ``name`` and implement :meth:`run`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this stage (e.g. 'collect', 'generate')."""
        ...

    @abstractmethod
    async def run(self, context: PipelineContext) -> PipelineContext:
        """Execute the stage, mutate *context*, and return it."""
        ...

    async def validate(self, context: PipelineContext) -> bool:
        """Optional pre-flight check. Return False to skip the stage."""
        return True

    def __repr__(self) -> str:
        return f"<Stage: {self.name}>"
