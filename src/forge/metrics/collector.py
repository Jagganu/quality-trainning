"""Central metrics collector — aggregates counters, gauges, timers, and costs."""

from __future__ import annotations

from forge.core.models import CostReport, DatasetMetrics, DeduplicationReport, DiversityScore, VerificationReport
from forge.metrics.counters import Counter, Gauge, Timer


class MetricsCollector:
    """One collector per pipeline run. Tracks everything."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._timers: dict[str, Timer] = {}
        self._costs: list[dict] = []

    # -- counters --
    def increment(self, name: str, n: int = 1) -> None:
        if name not in self._counters:
            self._counters[name] = Counter(name)
        self._counters[name].increment(n)

    def counter_value(self, name: str) -> int:
        return self._counters[name].value if name in self._counters else 0

    # -- gauges --
    def gauge(self, name: str, value: float) -> None:
        if name not in self._gauges:
            self._gauges[name] = Gauge(name)
        self._gauges[name].set(value)

    # -- timers --
    def timer(self, name: str) -> Timer:
        if name not in self._timers:
            self._timers[name] = Timer(name)
        return self._timers[name]

    # -- costs --
    def record_cost(self, model: str, tokens_in: int, tokens_out: int, cost: float) -> None:
        self._costs.append({"model": model, "tokens_in": tokens_in, "tokens_out": tokens_out, "cost": cost})

    # -- snapshot --
    def snapshot(self) -> DatasetMetrics:
        """Build a DatasetMetrics from current state."""
        total_cost = sum(c["cost"] for c in self._costs)
        total_in = sum(c["tokens_in"] for c in self._costs)
        total_out = sum(c["tokens_out"] for c in self._costs)

        cost_by_model: dict[str, float] = {}
        for c in self._costs:
            cost_by_model[c["model"]] = cost_by_model.get(c["model"], 0.0) + c["cost"]

        stage_durations = {name: t.elapsed for name, t in self._timers.items()}

        return DatasetMetrics(
            total_samples=self.counter_value("samples_generated"),
            verified_samples=self.counter_value("samples_verified"),
            rejected_samples=self.counter_value("samples_rejected"),
            cost_report=CostReport(
                total_cost=total_cost,
                cost_by_model=cost_by_model,
                total_tokens_in=total_in,
                total_tokens_out=total_out,
            ),
            stage_durations=stage_durations,
        )

    def reset(self) -> None:
        self._counters.clear()
        self._gauges.clear()
        self._timers.clear()
        self._costs.clear()
