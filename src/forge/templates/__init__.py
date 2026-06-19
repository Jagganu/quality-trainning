"""Built-in templates registry."""

from forge.templates.base import Template
from forge.templates.coding import CodingTemplate
from forge.templates.cybersecurity import CybersecurityTemplate
from forge.templates.reasoning import ReasoningTemplate

TEMPLATES: dict[str, type[Template]] = {
    "cybersecurity": CybersecurityTemplate,
    "reasoning": ReasoningTemplate,
    "coding": CodingTemplate,
}

__all__ = [
    "TEMPLATES",
    "Template",
    "CybersecurityTemplate",
    "ReasoningTemplate",
    "CodingTemplate",
]
