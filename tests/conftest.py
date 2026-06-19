"""Shared test fixtures for ForgeGravity."""

import pytest

from forge.core.config import ForgeSettings
from forge.core.models import (
    Document,
    Sample,
    SampleLineage,
)


@pytest.fixture
def settings():
    """Minimal ForgeSettings for testing (no real API keys)."""
    return ForgeSettings(
        default_model="gpt-4o-mini",
        output_dir="/tmp/forge_test",
        openai_api_key="test-key",
    )


@pytest.fixture
def sample():
    """A realistic sample for testing verification, export, etc."""
    return Sample(
        lineage=SampleLineage(
            template="cybersecurity",
            format="reasoning",
            pipeline_run_id="test-run-001",
            generation_model="gpt-4o-mini",
        ),
        content={
            "question": "What is cross-site scripting (XSS)?",
            "analysis": (
                "Cross-site scripting (XSS) is a web security vulnerability that allows "
                "an attacker to inject malicious scripts into web pages viewed by other users. "
                "There are three main types: Stored XSS, Reflected XSS, and DOM-based XSS. "
                "The attack works by exploiting insufficient input validation and output encoding."
            ),
            "answer": (
                "XSS is a security vulnerability where malicious scripts are injected into "
                "web applications. Prevention includes input validation, output encoding, "
                "Content Security Policy headers, and using templating engines with auto-escaping."
            ),
            "metadata": {
                "difficulty": "intermediate",
                "subtopic": "xss",
                "approach": "step_by_step",
                "confidence": 0.92,
            },
        },
    )


@pytest.fixture
def document():
    """A raw document for testing collect, verification, etc."""
    return Document(
        url="https://owasp.org/xss",
        title="Cross-Site Scripting Guide",
        content=(
            "Cross-site scripting (XSS) is a web security vulnerability that allows an "
            "attacker to inject malicious scripts into web pages viewed by other users. "
            "SQL injection attacks target databases by inserting malicious SQL code. "
            "Buffer overflow occurs when data exceeds the allocated memory buffer."
        ),
    )


@pytest.fixture
def sample_records():
    """A list of sample dicts in reasoning format for dataset testing."""
    return [
        {
            "question": "What is SQL injection?",
            "analysis": "SQL injection is a code injection technique...",
            "answer": "SQL injection allows attackers to...",
            "metadata": {"difficulty": "beginner"},
        },
        {
            "question": "Explain buffer overflow vulnerabilities",
            "analysis": "A buffer overflow occurs when...",
            "answer": "Buffer overflows can be mitigated by...",
            "metadata": {"difficulty": "advanced"},
        },
    ]
