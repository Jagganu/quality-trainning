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
        return "coding"

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
            "Generate training samples that develop software engineering skills.\n"
            "- Present realistic coding problems, not toy examples\n"
            "- Include actual code snippets in relevant languages (Python, JS, Go, Rust)\n"
            "- The investigation field should show systematic debugging approach\n"
            "- Patches should be minimal, correct, and well-explained\n"
            "- Verification should include test cases and edge cases\n"
            "- Cover the full spectrum: from algorithms to system design to debugging\n"
            "- Include performance considerations and Big-O analysis where relevant\n"
        )

    def estimated_page_count(self) -> int:
        return 40

    def estimated_sample_count(self) -> int:
        return 150
