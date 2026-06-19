"""Schema-level and content-level validation for raw sample dicts.

:class:`SampleValidator` operates on plain dictionaries so it can be
used early in the pipeline — before samples are parsed into Pydantic
models — as well as on already-validated data.
"""

from __future__ import annotations

from forge.utils.logging import get_logger

logger = get_logger(__name__)

_MIN_ANSWER_LENGTH = 20


class SampleValidator:
    """Validates raw sample dictionaries for completeness and quality."""

    @staticmethod
    def validate_schema(sample: dict, required_keys: list[str]) -> list[str]:
        """Return a list of *required_keys* missing from *sample*.

        Parameters
        ----------
        sample:
            The raw sample dictionary to check.
        required_keys:
            Keys that must be present in *sample*.

        Returns
        -------
        list[str]
            Names of missing keys (empty if all present).
        """
        return [k for k in required_keys if k not in sample]

    @staticmethod
    def validate_content(sample: dict) -> list[str]:
        """Check for empty or suspiciously short content values.

        Rules
        -----
        * String values must not be empty.
        * Any ``answer``, ``result``, or ``analysis`` field must be at
          least :data:`_MIN_ANSWER_LENGTH` characters.

        Returns
        -------
        list[str]
            Descriptions of each issue found.
        """
        issues: list[str] = []
        for key, value in sample.items():
            if isinstance(value, str) and value.strip() == "":
                issues.append(f"Key '{key}' is an empty string")

        for key in ("answer", "result", "analysis"):
            val = sample.get(key)
            if isinstance(val, str) and 0 < len(val.strip()) < _MIN_ANSWER_LENGTH:
                issues.append(
                    f"Key '{key}' is too short ({len(val.strip())} chars, "
                    f"minimum {_MIN_ANSWER_LENGTH})"
                )

        return issues

    @staticmethod
    def validate_lineage(sample: dict) -> bool:
        """Check that core lineage fields are present and non-empty.

        The following keys are required: ``sample_id``,
        ``source_documents``, ``generation_model``.

        Returns
        -------
        bool
            ``True`` if all lineage fields are present and truthy.
        """
        required = ("sample_id", "source_documents", "generation_model")
        lineage = sample.get("lineage", sample)
        return all(bool(lineage.get(k)) for k in required)

    @staticmethod
    def validate_all(
        sample: dict,
        required_keys: list[str],
    ) -> tuple[bool, list[str]]:
        """Run all validations and return an aggregate result.

        Parameters
        ----------
        sample:
            The raw sample dictionary to check.
        required_keys:
            Keys that must be present in the sample *content*.

        Returns
        -------
        tuple[bool, list[str]]
            ``(passed, issues)`` where *passed* is ``True`` only if
            every individual check passed.
        """
        issues: list[str] = []

        missing = SampleValidator.validate_schema(sample, required_keys)
        if missing:
            issues.extend(f"Missing key: {k}" for k in missing)

        content_issues = SampleValidator.validate_content(sample)
        issues.extend(content_issues)

        if not SampleValidator.validate_lineage(sample):
            issues.append("Lineage validation failed: missing sample_id, source_documents, or generation_model")

        passed = len(issues) == 0
        if not passed:
            logger.debug("Sample validation failed with %d issues", len(issues))
        return passed, issues
