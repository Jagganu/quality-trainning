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
        return "principles"

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
            "Generate training samples that teach cybersecurity PRINCIPLES and the 'WHY' behind them.\n"
            "- Each sample must identify the FUNDAMENTAL SECURITY PRINCIPLE (e.g., least privilege, defense in depth, fail-safe defaults, zero trust)\n"
            "- Explain WHY the principle exists: the underlying threat model and attack surface it addresses\n"
            "- Show the VIOLATION: how ignoring the principle leads to specific vulnerability classes\n"
            "- Demonstrate the CORRECT APPLICATION: secure patterns that embody the principle\n"
            "- Define BOUNDARY CONDITIONS: when the principle might be relaxed and the compensating controls needed\n"
            "- Use real CVEs as case studies of principle violations, not just exploit demonstrations\n"
            "- Include difficulty levels from beginner (recognizing the principle) to advanced (novel principle compositions)\n"
            "- Focus on TRANSFERABLE UNDERSTANDING, not memorizing specific exploits\n"
        )

    def estimated_page_count(self) -> int:
        return 50

    def estimated_sample_count(self) -> int:
        return 200
