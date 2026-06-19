"""Pydantic Settings for ForgeGravity — loads from forge.toml, .env, and env vars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CollectSettings(BaseSettings):
    """Settings for the Collect stage."""
    max_pages: int = 50
    max_documents: int = 20
    requests_per_second: float = 2.0
    respect_robots_txt: bool = True
    user_agent: str = "ForgeGravity/0.1 (dataset-builder)"
    blocked_domains: list[str] = Field(default_factory=lambda: [
        "pinterest.com", "reddit.com", "facebook.com", "twitter.com",
        "instagram.com", "tiktok.com", "youtube.com", "linkedin.com",
        "quora.com", "medium.com", "substack.com",
        "wikihow.com", "ehow.com", "livestrong.com",
        "answerbag.com", "blurtit.com",
    ])
    min_word_count: int = 100
    min_content_length: int = 200


class CleanSettings(BaseSettings):
    """Settings for the Clean stage."""
    min_content_length: int = 200
    max_chunk_size: int = 4000
    min_quality_score: float = 0.3
    remove_boilerplate: bool = True
    language: str = "en"
    min_avg_sentence_length: int = 5
    max_code_block_ratio: float = 0.4
    min_char_entropy: float = 3.0


class GenerateSettings(BaseSettings):
    """Settings for the Generate stage."""
    default_format: str = "reasoning"
    max_samples: int = 100
    batch_size: int = 10
    max_concurrent: int = 5
    temperature: float = 0.7


class VerifySettings(BaseSettings):
    """Settings for the Verify stage."""
    enabled: bool = True
    critic_model: str = ""
    scorer_model: str = ""
    min_pass_rate: float = 0.6
    min_score: float = 0.7


class BudgetSettings(BaseSettings):
    """Settings for cost budget enforcement."""
    max_cost_usd: float | None = None
    warn_at_percent: float = 80.0


class QualityGateSettings(BaseSettings):
    """Configurable thresholds for quality gates."""
    max_duplicate_rate: float = 0.10
    min_diversity_score: float = 0.5
    min_verification_score: float = 0.7


class ForgeSettings(BaseSettings):
    """Root configuration — all settings flow from here.

    Loading priority (highest to lowest):
        1. Environment variables (FORGE_*)
        2. .env file
        3. forge.toml
        4. Defaults in this class
    """
    model_config = SettingsConfigDict(
        env_prefix="FORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    output_dir: str = "./output"
    default_provider: str = "openai"
    default_model: str = "gpt-4o-mini"
    log_level: str = "INFO"

    # API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Sub-settings (populated from forge.toml sections)
    collect: CollectSettings = Field(default_factory=CollectSettings)
    clean: CleanSettings = Field(default_factory=CleanSettings)
    generate: GenerateSettings = Field(default_factory=GenerateSettings)
    verify: VerifySettings = Field(default_factory=VerifySettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    quality_gates: QualityGateSettings = Field(default_factory=QualityGateSettings)


def load_settings(config_path: str | Path | None = None) -> ForgeSettings:
    """Load settings, optionally from a specific config file path.

    If ``config_path`` is provided and the file exists, its TOML content
    is merged into settings. Otherwise, defaults + env vars are used.
    """
    import tomllib

    overrides: dict[str, Any] = {}

    path = Path(config_path) if config_path else Path("forge.toml")
    if path.exists():
        with open(path, "rb") as f:
            toml_data = tomllib.load(f)

        # Flatten TOML sections into overrides
        for section in ("collect", "clean", "generate", "verify", "budget", "quality_gates"):
            if section in toml_data:
                overrides[section] = toml_data[section]

        general = toml_data.get("general", {})
        overrides.update(general)

    return ForgeSettings(**overrides)
