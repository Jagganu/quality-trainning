"""Tests for forge.datasets.validators — dataset-level quality checks.

Exercises every public method of DatasetValidator (validate_completeness,
validate_consistency, validate_lineage, full_validation) including edge
cases like empty inputs, mixed valid/invalid records, and interaction
between the sub-validators when combined in full_validation.
"""

from __future__ import annotations

import pytest

from forge.datasets.validators import DatasetValidator, ValidationReport


@pytest.fixture
def validator():
    return DatasetValidator()


# ── validate_completeness ─────────────────────────────────────────────────

class TestValidateCompleteness:
    """Records must be non-empty and have string fields ≥ 5 characters."""

    def test_valid_record(self, validator):
        records = [{"question": "What is a buffer overflow?", "answer": "A buffer overflow is..."}]
        assert validator.validate_completeness(records) == []

    def test_empty_record(self, validator):
        errors = validator.validate_completeness([{}])
        assert len(errors) == 1
        assert "empty record" in errors[0]

    def test_short_string_field(self, validator):
        """String fields under 5 characters should be flagged."""
        records = [{"question": "Hi", "answer": "Yes"}]
        errors = validator.validate_completeness(records)
        assert len(errors) == 2
        assert all("too short" in e for e in errors)

    def test_exactly_five_chars_passes(self, validator):
        """A field with exactly 5 characters should pass."""
        records = [{"question": "Hello"}]
        assert validator.validate_completeness(records) == []

    def test_non_string_fields_ignored(self, validator):
        """Non-string values (ints, dicts, lists) should not be length-checked."""
        records = [{"count": 1, "tags": ["a"], "meta": {"k": "v"}}]
        assert validator.validate_completeness(records) == []

    def test_metadata_key_exempt(self, validator):
        """The 'metadata' key is explicitly exempt from the length check."""
        records = [{"metadata": "ok"}]  # "ok" is only 2 chars
        assert validator.validate_completeness(records) == []

    def test_multiple_records_indexed(self, validator):
        """Errors must reference correct record indices."""
        records = [
            {"question": "Long enough question here"},
            {},
            {"answer": "No"},
        ]
        errors = validator.validate_completeness(records)
        assert any("Record 1" in e and "empty" in e for e in errors)
        assert any("Record 2" in e and "too short" in e for e in errors)

    def test_whitespace_only_flagged(self, validator):
        """Whitespace-only strings with len < 5 after strip should be caught."""
        records = [{"question": "    "}]
        errors = validator.validate_completeness(records)
        # "    " is 4 chars, strip() gives "", which has len 0 < 5
        assert len(errors) == 1
        assert "too short" in errors[0]


# ── validate_consistency ──────────────────────────────────────────────────

class TestValidateConsistency:
    """All records should share the same set of top-level keys."""

    def test_consistent_records(self, validator):
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        assert validator.validate_consistency(records) == []

    def test_empty_list(self, validator):
        assert validator.validate_consistency([]) == []

    def test_single_record(self, validator):
        """A single record is always consistent with itself."""
        assert validator.validate_consistency([{"x": 1}]) == []

    def test_extra_keys(self, validator):
        records = [{"a": 1}, {"a": 1, "b": 2}]
        errors = validator.validate_consistency(records)
        assert len(errors) == 1
        assert "extra keys" in errors[0]

    def test_missing_keys(self, validator):
        records = [{"a": 1, "b": 2}, {"a": 1}]
        errors = validator.validate_consistency(records)
        assert len(errors) == 1
        assert "missing keys" in errors[0]

    def test_both_extra_and_missing(self, validator):
        """A record with both extra and missing keys gets two errors."""
        records = [{"a": 1, "b": 2}, {"a": 1, "c": 3}]
        errors = validator.validate_consistency(records)
        assert len(errors) == 2

    def test_uses_first_record_as_reference(self, validator):
        """Only the first record defines the reference schema."""
        records = [
            {"x": 1},
            {"x": 1, "y": 2},
            {"x": 1, "y": 2},
        ]
        errors = validator.validate_consistency(records)
        # Records 1 and 2 both have extra key "y"
        assert len(errors) == 2


# ── validate_lineage ──────────────────────────────────────────────────────

class TestValidateLineage:
    """The 'metadata' field should be a dict when present."""

    def test_dict_metadata_passes(self, validator):
        records = [{"metadata": {"run": "123"}}]
        assert validator.validate_lineage(records) == []

    def test_missing_metadata_passes(self, validator):
        """Records without 'metadata' should be fine (default is empty dict)."""
        records = [{"question": "Q?"}]
        assert validator.validate_lineage(records) == []

    def test_non_dict_metadata_error(self, validator):
        records = [{"metadata": "not a dict"}]
        errors = validator.validate_lineage(records)
        assert len(errors) == 1
        assert "not a dict" in errors[0]

    def test_list_metadata_error(self, validator):
        records = [{"metadata": [1, 2, 3]}]
        errors = validator.validate_lineage(records)
        assert len(errors) == 1


# ── full_validation ───────────────────────────────────────────────────────

class TestFullValidation:
    """Integration of schema + completeness + consistency + lineage."""

    def test_perfect_records(self, validator, sample_records):
        """The conftest sample_records fixture should be fully valid."""
        report = validator.full_validation(sample_records, "reasoning")

        assert isinstance(report, ValidationReport)
        assert report.total_records == 2
        assert report.valid_records == 2
        assert report.pass_rate == 1.0
        assert report.errors == []

    def test_empty_dataset(self, validator):
        report = validator.full_validation([], "reasoning")
        assert report.total_records == 0
        assert report.pass_rate == 0.0

    def test_schema_errors_counted(self, validator):
        """Records missing required keys should reduce valid_records."""
        records = [
            {"question": "Good enough question"},  # missing analysis, answer, metadata
        ]
        report = validator.full_validation(records, "reasoning")
        assert report.total_records == 1
        assert report.valid_records == 0
        assert report.pass_rate == 0.0
        assert len(report.errors) >= 3  # 3 missing keys + potential completeness

    def test_mixed_valid_and_invalid(self, validator):
        """Report should correctly count when some records pass and some fail."""
        records = [
            {
                "question": "Valid long question text",
                "analysis": "Valid long analysis text",
                "answer": "Valid long answer text",
                "metadata": {"difficulty": "beginner"},
            },
            {
                "question": "Another valid question",
                "analysis": "Another valid analysis",
                "answer": "Another valid answer",
                "metadata": {"difficulty": "advanced"},
            },
            {"question": "Missing fields"},  # fails schema
        ]
        report = validator.full_validation(records, "reasoning")
        assert report.total_records == 3
        assert report.valid_records == 2
        assert 0.5 < report.pass_rate < 1.0

    def test_lineage_issues_are_warnings_not_errors(self, validator):
        """validate_lineage results go into warnings, not errors."""
        records = [{
            "question": "Valid question text here",
            "analysis": "Valid analysis text here",
            "answer": "Valid answer text here",
            "metadata": "not_a_dict",  # lineage warning
        }]
        report = validator.full_validation(records, "reasoning")
        assert any("not a dict" in w for w in report.warnings)

    def test_pass_rate_is_rounded(self, validator):
        """pass_rate should be rounded to 4 decimal places."""
        records = [
            {"question": "Valid", "analysis": "Valid", "answer": "Valid", "metadata": {}},
            {"question": "Valid", "analysis": "Valid", "answer": "Valid", "metadata": {}},
            {"broken": "record"},
        ]
        report = validator.full_validation(records, "reasoning")
        # 2/3 = 0.666... should be rounded
        assert report.pass_rate == round(report.pass_rate, 4)
