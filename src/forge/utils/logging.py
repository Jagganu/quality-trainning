"""Structured logging with Rich integration."""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)
_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with Rich handler."""
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=_console,
                rich_tracebacks=True,
                show_path=False,
                markup=True,
            )
        ],
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Configures Rich handler on first call."""
    setup_logging()
    return logging.getLogger(name)


def get_console() -> Console:
    """Get the shared Rich console for direct output."""
    return _console
