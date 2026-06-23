"""Programmatic ground-truth checks for generated candidates.

These checks never call an LLM — they execute code against test cases or
evaluate math symbolically — so they're free and unambiguous wherever they
apply. They should always run *before* any judge/scorer call: a candidate
that fails a programmatic check can be dropped immediately without paying
for an LLM opinion on it.

Security note
--------------
``CodeChecker`` executes model-generated Python in a subprocess. This is
**process isolation with a timeout, not a full sandbox** — it does not
block filesystem or network access. Do not run it against untrusted code
on a machine with secrets, credentials, or sensitive data reachable from
the process environment. For production use, run it in a disposable
container or VM with no network egress instead.
"""

from __future__ import annotations

import asyncio
import re
import sys
import tempfile
from pathlib import Path

from forge.utils.logging import get_logger

logger = get_logger(__name__)


class CodeChecker:
    """Executes generated code against test cases via subprocess.

    Expects the coding-format sample content to provide a ``patch`` (the
    code under test) and a ``verification`` field containing test code or
    assertions appended after the patch. Returns True/False/None (None =
    couldn't determine, e.g. no test code present — falls back to LLM
    judging for that candidate).
    """

    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def check(self, code: str, test_code: str) -> bool | None:
        """Run *code* followed by *test_code* in a subprocess. Returns
        True if it exits zero, False if it raises/exits non-zero, None if
        there's nothing to test.
        """
        if not test_code or not test_code.strip():
            return None

        script = f"{code}\n\n{test_code}\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            script_path = Path(f.name)

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_seconds
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                logger.debug("CodeChecker timed out after %.1fs", self.timeout_seconds)
                return False

            if proc.returncode != 0:
                logger.debug("CodeChecker failed: %s", stderr.decode(errors="replace")[:500])
                return False
            return True
        except Exception as exc:
            logger.warning("CodeChecker subprocess error: %s", exc)
            return None
        finally:
            script_path.unlink(missing_ok=True)


class MathChecker:
    """Symbolic/numeric equality check for math answers using SymPy.

    SymPy is an optional dependency — if it isn't installed, ``check``
    always returns None (defer to LLM/clustering instead of failing the
    pipeline).
    """

    def check(self, candidate_answer: str, reference_answer: str | None) -> bool | None:
        if not reference_answer:
            return None
        try:
            import sympy
            from sympy.parsing.sympy_parser import parse_expr
        except ImportError:
            logger.debug("sympy not installed — skipping symbolic math check")
            return None

        try:
            lhs = parse_expr(self._extract_expression(candidate_answer))
            rhs = parse_expr(self._extract_expression(reference_answer))
            diff = sympy.simplify(lhs - rhs)
            return bool(diff == 0)
        except Exception:
            return None

    @staticmethod
    def _extract_expression(text: str) -> str:
        """Pull the last numeric/algebraic expression out of free text."""
        match = re.search(r"[-+]?\d*\.?\d+(?:/\d+)?", text)
        return match.group(0) if match else text.strip()
