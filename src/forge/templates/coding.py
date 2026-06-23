"""Coding template — software engineering, debugging, and algorithms."""

from __future__ import annotations

from forge.templates.base import Template


class CodingTemplate(Template):
    """Template for software engineering, debugging, and code review."""

    @property
    def name(self) -> str:
        return "coding"

    @property
    def description(self) -> str:
        return "Software engineering, debugging, code review, and algorithms"

    @property
    def default_format(self) -> str:
        return "principles"

    def seed_topics(self) -> list[str]:
        return [
            "algorithm design and complexity analysis",
            "data structures implementation and trade-offs",
            "debugging techniques for production systems",
            "code review best practices and anti-patterns",
            "design patterns Gang of Four practical usage",
            "database query optimization SQL performance",
            "concurrency and parallel programming patterns",
            "API design REST GraphQL best practices",
            "testing strategies unit integration end-to-end",
            "refactoring legacy code safely",
            "memory management and performance profiling",
            "distributed systems design patterns",
            "error handling and resilience patterns",
            "CI/CD pipeline configuration and optimization",
        ]

    def generation_guidelines(self) -> str:
        return (
            "Generate training samples that teach SOFTWARE ENGINEERING PRINCIPLES and the 'WHY' behind them.\n"
            "- Each sample must identify the FUNDAMENTAL ENGINEERING PRINCIPLE (e.g., SOLID, DRY, KISS, YAGNI, separation of concerns, single responsibility, loose coupling, high cohesion, fail fast, design for failure)\n"
            "- Explain WHY the principle exists: the underlying complexity or failure mode it prevents\n"
            "- Show VIOLATION: real-world technical debt, bugs, or scaling failures from ignoring the principle\n"
            "- Demonstrate CORRECT APPLICATION: clean code, architecture, or design pattern embodying the principle\n"
            "- Define BOUNDARY CONDITIONS: when pragmatic trade-offs justify bending the principle (with mitigations)\n"
            "- Cover algorithms, system design, debugging, and code review through a principles lens\n"
            "- Focus on TRANSFERABLE ENGINEERING JUDGMENT, not syntax memorization\n"
        )

    def estimated_page_count(self) -> int:
        return 40

    def estimated_sample_count(self) -> int:
        return 150
