"""Tests for core.config — settings loading and defaults."""

from forge.core.config import ForgeSettings, load_settings


def test_default_settings():
    """ForgeSettings should have sensible defaults."""
    s = ForgeSettings()
    assert s.default_model == "gpt-4o-mini"
    assert s.output_dir == "./output"
    assert s.log_level == "INFO"
    assert s.collect.max_pages == 50
    assert s.generate.default_format == "reasoning"
    assert s.verify.enabled is True
    assert s.quality_gates.max_duplicate_rate == 0.10


def test_settings_override():
    """Field overrides should apply."""
    s = ForgeSettings(default_model="claude-3-haiku", output_dir="/custom")
    assert s.default_model == "claude-3-haiku"
    assert s.output_dir == "/custom"


def test_load_settings_no_config():
    """load_settings without a file should return defaults."""
    s = load_settings(config_path="/nonexistent/path/forge.toml")
    assert isinstance(s, ForgeSettings)
    assert s.default_model == "gpt-4o-mini"


def test_sub_settings_defaults():
    """Sub-settings should have independent defaults."""
    s = ForgeSettings()
    assert s.budget.max_cost_usd is None
    assert s.budget.warn_at_percent == 80.0
    assert s.verify.critic_model == "gpt-4o-mini"
    assert s.verify.min_score == 0.7
