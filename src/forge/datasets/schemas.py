"""Dataset format schemas - validates records against known structures."""

from __future__ import annotations

from typing import Any

SCHEMAS: dict[str, dict[str, Any]] = {
    "reasoning": {
        "required": ["question", "analysis", "answer", "metadata"],
        "properties": {
            "question": {"type": "string"},
            "analysis": {"type": "string"},
            "answer": {"type": "string"},
            "metadata": {"type": "object"},
        },
    },
    "instruction": {
        "required": ["messages"],
        "properties": {
            "messages": {"type": "array"},
        },
    },
    "agent": {
        "required": ["observation", "thought", "action", "result"],
        "properties": {
            "observation": {"type": "string"},
            "thought": {"type": "string"},
            "action": {"type": "string"},
            "result": {"type": "string"},
        },
    },
    "coding": {
        "required": ["issue", "investigation", "patch", "verification"],
        "properties": {
            "issue": {"type": "string"},
            "investigation": {"type": "string"},
            "patch": {"type": "string"},
            "verification": {"type": "string"},
        },
    },
    "chat": {
        "required": ["conversations"],
        "properties": {
            "conversations": {"type": "array"},
        },
    },
}


class DatasetSchema:
    """Schema definitions and validation for dataset formats."""

    @staticmethod
    def for_format(format_name: str) -> dict[str, Any]:
        """Return the JSON-schema-like dict for *format_name*."""
        schema = SCHEMAS.get(format_name)
        if schema is None:
            raise ValueError(f"Unknown format '{format_name}'. Available: {list(SCHEMAS)}")
        return schema

    @staticmethod
    def validate(records: list[dict], format_name: str) -> list[str]:
        """Validate *records* against *format_name* schema, return errors."""
        schema = SCHEMAS.get(format_name)
        if schema is None:
            return [f"Unknown format: {format_name}"]

        required = schema.get("required", [])
        errors: list[str] = []

        for idx, record in enumerate(records):
            for key in required:
                if key not in record:
                    errors.append(f"Record {idx}: missing required key '{key}'")
                elif DatasetSchema._is_empty_value(record[key]):
                    errors.append(f"Record {idx}: key '{key}' is empty")

        return errors

    @staticmethod
    def available_formats() -> list[str]:
        """Return names of all supported formats."""
        return list(SCHEMAS.keys())

    @staticmethod
    def _is_empty_value(value: Any) -> bool:
        """Return whether a required value is present but empty."""
        if value == 0:
            return False
        if value is None:
            return True
        if isinstance(value, str):
            return value == ""
        if isinstance(value, (list, tuple, set)):
            return len(value) == 0
        return False
