"""Thread-safe counters, gauges, and timers for metrics tracking."""

from __future__ import annotations

import threading
import time


class Counter:
    """Thread-safe integer counter."""

    def __init__(self, name: str, value: int = 0) -> None:
        self.name = name
        self._value = value
        self._lock = threading.Lock()

    @property
    def value(self) -> int:
        return self._value

    def increment(self, n: int = 1) -> None:
        with self._lock:
            self._value += n

    def decrement(self, n: int = 1) -> None:
        with self._lock:
            self._value -= n

    def reset(self) -> None:
        with self._lock:
            self._value = 0

    def __repr__(self) -> str:
        return f"Counter({self.name}={self._value})"


class Gauge:
    """Thread-safe float gauge."""

    def __init__(self, name: str, value: float = 0.0) -> None:
        self.name = name
        self._value = value
        self._lock = threading.Lock()

    @property
    def value(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def increment(self, n: float = 1.0) -> None:
        with self._lock:
            self._value += n

    def decrement(self, n: float = 1.0) -> None:
        with self._lock:
            self._value -= n

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0

    def __repr__(self) -> str:
        return f"Gauge({self.name}={self._value:.4f})"


class Timer:
    """Context-manager timer that records elapsed seconds."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._start: float | None = None
        self._elapsed: float = 0.0

    @property
    def elapsed(self) -> float:
        if self._start is not None:
            return time.monotonic() - self._start
        return self._elapsed

    @property
    def running(self) -> bool:
        return self._start is not None

    def __enter__(self) -> Timer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._start is not None:
            self._elapsed = time.monotonic() - self._start
            self._start = None

    def __repr__(self) -> str:
        return f"Timer({self.name}={self.elapsed:.3f}s)"
