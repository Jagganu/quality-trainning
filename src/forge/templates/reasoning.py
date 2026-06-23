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
        return "principles"

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
            "Generate training samples that teach REASONING PRINCIPLES and the 'WHY' behind them.\n"
            "- Each sample must identify the FUNDAMENTAL REASONING PRINCIPLE (e.g., Occam's razor, Bayesian updating, reductio ad absurdum, first principles thinking, falsifiability, modular decomposition)\n"
            "- Explain WHY the principle works: the mathematical/epistemological foundation\n"
            "- Show MISAPPLICATION: common cognitive biases or logical fallacies when the principle is violated\n"
            "- Demonstrate CORRECT APPLICATION: step-by-step use of the principle to solve the problem\n"
            "- Define BOUNDARY CONDITIONS: when the principle breaks down or needs adaptation\n"
            "- Use diverse domains (math, logic, science, ethics, systems) to show principle transferability\n"
            "- Focus on DEEP UNDERSTANDING of the reasoning process, not just getting the right answer\n"
        )

    def estimated_page_count(self) -> int:
        return 30

    def estimated_sample_count(self) -> int:
        return 100
