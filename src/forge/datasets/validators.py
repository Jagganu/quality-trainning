"""Dataset-level validators — checks completeness, consistency, and lineage."""

from __future__ import annotations

from pydantic import BaseModel, Field

from forge.datasets.schemas import DatasetSchema


class ValidationReport(BaseModel):
    """Summary of dataset validation."""

    total_records: int = 0
    valid_records: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pass_rate: float = 0.0


class DatasetValidator:
    """Validates entire datasets (not individual samples)."""

    def validate_completeness(self, records: list[dict]) -> list[str]:
        """Check that no records are empty or trivially short."""
        errors: list[str] = []
        for idx, record in enumerate(records):
            if not record:
                errors.append(f"Record {idx}: empty record")
                continue
            # Check that string values have meaningful content
            for key, val in record.items():
                if isinstance(val, str) and len(val.strip()) < 5 and key != "metadata":
                    errors.append(f"Record {idx}: field '{key}' is too short ({len(val)} chars)")
        return errors

    def validate_consistency(self, records: list[dict]) -> list[str]:
        """Check that all records share the same set of top-level keys."""
        if not records:
            return []

        reference_keys = set(records[0].keys())
        errors: list[str] = []
        for idx, record in enumerate(records[1:], start=1):
            keys = set(record.keys())
            missing = reference_keys - keys
            extra = keys - reference_keys
            if missing:
                errors.append(f"Record {idx}: missing keys {missing} vs record 0")
            if extra:
                errors.append(f"Record {idx}: extra keys {extra} vs record 0")
        return errors

    def validate_lineage(self, records: list[dict]) -> list[str]:
        """Check that lineage metadata is present where expected."""
        errors: list[str] = []
        for idx, record in enumerate(records):
            meta = record.get("metadata", {})
            if not isinstance(meta, dict):
                errors.append(f"Record {idx}: 'metadata' is not a dict")
        return errors

    def full_validation(
        self,
        records: list[dict],
        format_name: str,
    ) -> ValidationReport:
        """Run all validations and return a combined report."""
        schema_errors = DatasetSchema.validate(records, format_name)
        completeness_errors = self.validate_completeness(records)
        consistency_errors = self.validate_consistency(records)
        lineage_warnings = self.validate_lineage(records)

        all_errors = schema_errors + completeness_errors + consistency_errors
        valid = len(records) - len({
            e.split(":")[0] for e in all_errors if e.startswith("Record ")
        })

        return ValidationReport(
            total_records=len(records),
            valid_records=max(0, valid),
            errors=all_errors,
            warnings=lineage_warnings,
            pass_rate=round(valid / len(records), 4) if records else 0.0,
        )
