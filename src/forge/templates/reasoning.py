"""Reasoning template — general logical reasoning and problem-solving."""

from __future__ import annotations

from forge.templates.base import Template


class ReasoningTemplate(Template):
    """Template for general reasoning, logic, and problem-solving."""

    @property
    def name(self) -> str:
        return "reasoning"

    @property
    def description(self) -> str:
        return "General reasoning and problem-solving across domains"

    @property
    def default_format(self) -> str:
        return "reasoning"

    def seed_topics(self) -> list[str]:
        return [
            "formal logic and propositional reasoning",
            "mathematical proof techniques and strategies",
            "critical thinking and argument analysis",
            "probability and statistical reasoning",
            "causal reasoning and counterfactuals",
            "analogical reasoning across domains",
            "spatial reasoning and geometric intuition",
            "constraint satisfaction and optimization",
            "game theory and strategic thinking",
            "scientific method and hypothesis testing",
            "ethical reasoning and moral dilemmas",
            "systems thinking and feedback loops",
        ]

    def generation_guidelines(self) -> str:
        return (
            "Generate training samples that develop strong reasoning skills.\n"
            "- Questions should require genuine multi-step reasoning\n"
            "- Include problems from diverse domains (math, logic, science, ethics)\n"
            "- The analysis field must show clear, step-by-step reasoning\n"
            "- Avoid trivial factoid questions — focus on understanding and application\n"
            "- Include edge cases and counterexamples where relevant\n"
            "- Vary problem structures: deductive, inductive, abductive reasoning\n"
        )

    def estimated_page_count(self) -> int:
        return 30

    def estimated_sample_count(self) -> int:
        return 100
