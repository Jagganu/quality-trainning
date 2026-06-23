"""Dataset format registry — maps format names to classes."""

from __future__ import annotations

from forge.stages.generate.formats.agent import AgentFormat
from forge.stages.generate.formats.base import DatasetFormat
from forge.stages.generate.formats.chat import ChatFormat
from forge.stages.generate.formats.coding import CodingFormat
from forge.stages.generate.formats.instruction import InstructionFormat
from forge.stages.generate.formats.principles import PrinciplesFormat
from forge.stages.generate.formats.reasoning import ReasoningFormat

FORMAT_REGISTRY: dict[str, type[DatasetFormat]] = {
    "reasoning": ReasoningFormat,
    "instruction": InstructionFormat,
    "agent": AgentFormat,
    "coding": CodingFormat,
    "chat": ChatFormat,
    "principles": PrinciplesFormat,
}

__all__ = [
    "FORMAT_REGISTRY",
    "DatasetFormat",
    "ReasoningFormat",
    "InstructionFormat",
    "AgentFormat",
    "CodingFormat",
    "ChatFormat",
]
