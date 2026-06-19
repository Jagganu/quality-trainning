"""Cybersecurity template — generates security-focused training data."""

from __future__ import annotations

from forge.templates.base import Template


class CybersecurityTemplate(Template):
    """Template for cybersecurity concepts, vulnerabilities, and secure coding."""

    @property
    def name(self) -> str:
        return "cybersecurity"

    @property
    def description(self) -> str:
        return "Cybersecurity concepts, vulnerabilities, CTF challenges, and secure coding"

    @property
    def default_format(self) -> str:
        return "reasoning"

    def seed_topics(self) -> list[str]:
        return [
            "SQL injection attack techniques and prevention",
            "cross-site scripting XSS vulnerability types",
            "buffer overflow exploitation and mitigations",
            "cross-site request forgery CSRF defense mechanisms",
            "server-side request forgery SSRF attacks",
            "insecure deserialization vulnerabilities",
            "authentication bypass techniques",
            "privilege escalation in Linux systems",
            "web application firewall evasion",
            "cryptographic implementation weaknesses",
            "API security best practices OWASP",
            "container security Docker Kubernetes",
            "network penetration testing methodology",
            "malware analysis reverse engineering fundamentals",
            "incident response and forensics procedures",
            "zero-day vulnerability discovery techniques",
            "secure code review SAST DAST",
        ]

    def generation_guidelines(self) -> str:
        return (
            "Generate training samples that teach cybersecurity concepts accurately.\n"
            "- Include realistic but SAFE examples (no actual exploit code for active systems)\n"
            "- Cover both offensive (red team) and defensive (blue team) perspectives\n"
            "- Reference real CVEs where relevant, but focus on understanding the concepts\n"
            "- Include difficulty levels from beginner (basic concepts) to advanced (novel attacks)\n"
            "- Questions should require multi-step reasoning, not simple recall\n"
            "- Analysis should explain the 'why' behind each vulnerability, not just the 'how'\n"
            "- Answers should include mitigation strategies and secure alternatives\n"
        )

    def estimated_page_count(self) -> int:
        return 50

    def estimated_sample_count(self) -> int:
        return 200
