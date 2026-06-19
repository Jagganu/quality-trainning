"""Terminal trajectory collector — records command-line interactions."""

from __future__ import annotations

import asyncio

from forge.utils.logging import get_logger

logger = get_logger(__name__)


class TerminalTrajectoryCollector:
    """Captures terminal command executions and their output."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def record_command(self, command: str) -> str:
        """Execute a shell command and return its output.

        Parameters
        ----------
        command:
            The shell command to execute.

        Returns
        -------
        str
            Combined stdout and stderr output, truncated to 10 000 characters.
        """
        logger.debug("Executing command: %s", command)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout,
            )

            output_parts: list[str] = []
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                output_parts.append(stderr.decode("utf-8", errors="replace"))

            output = "\n".join(output_parts)
            # Truncate very long outputs
            if len(output) > 10_000:
                output = output[:10_000] + "\n... [truncated]"

            logger.debug("Command exit code: %d", proc.returncode or 0)
            return output

        except asyncio.TimeoutError:
            logger.warning("Command timed out after %.0fs: %s", self._timeout, command)
            return f"[Timeout after {self._timeout}s]"
        except Exception as exc:
            logger.error("Command failed: %s — %s", command, exc)
            return f"[Error: {exc}]"
