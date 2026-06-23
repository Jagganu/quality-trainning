"""Tests for forge.verification.programmatic — CodeChecker (real subprocess
execution) and MathChecker (symbolic equality via sympy, skipped if sympy
isn't installed).
"""

from __future__ import annotations

import pytest

from forge.verification.programmatic import CodeChecker, MathChecker


class TestCodeChecker:
    @pytest.mark.asyncio
    async def test_passing_code_returns_true(self):
        checker = CodeChecker()
        result = await checker.check(
            "def add(a, b):\n    return a + b",
            "assert add(2, 3) == 5",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_failing_assertion_returns_false(self):
        checker = CodeChecker()
        result = await checker.check(
            "def add(a, b):\n    return a - b",
            "assert add(2, 3) == 5",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_syntax_error_returns_false(self):
        checker = CodeChecker()
        result = await checker.check("def broken(:\n    pass", "assert True")
        assert result is False

    @pytest.mark.asyncio
    async def test_no_test_code_returns_none(self):
        checker = CodeChecker()
        result = await checker.check("def f(): return 1", "")
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_false(self):
        checker = CodeChecker(timeout_seconds=0.3)
        result = await checker.check("import time\ntime.sleep(10)", "assert True")
        assert result is False


class TestMathChecker:
    def test_no_reference_returns_none(self):
        checker = MathChecker()
        assert checker.check("42", None) is None

    def test_matching_values_return_true_when_sympy_available(self):
        pytest.importorskip("sympy")
        checker = MathChecker()
        result = checker.check("The answer is 42.", "42")
        assert result is True

    def test_mismatched_values_return_false_when_sympy_available(self):
        pytest.importorskip("sympy")
        checker = MathChecker()
        result = checker.check("The answer is 7.", "42")
        assert result is False
