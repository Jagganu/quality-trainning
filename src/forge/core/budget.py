"""Cost budgeting — tracks spend and enforces hard limits."""

from __future__ import annotations

from forge.core.models import CostEstimate, CostReport
from forge.utils.cost import estimate_cost
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class BudgetExceededError(Exception):
    """Raised when spending exceeds the configured limit."""

    def __init__(self, spent: float, limit: float, stage: str = "") -> None:
        self.spent = spent
        self.limit = limit
        self.stage = stage
        super().__init__(
            f"Budget exceeded: ${spent:.4f} spent of ${limit:.2f} limit"
            + (f" (during {stage})" if stage else "")
        )


class CostBudget:
    """Tracks token usage and cost, enforces spending limits."""

    def __init__(self, max_cost: float | None = None) -> None:
        self.max_cost = max_cost
        self.current_cost: float = 0.0
        self._cost_by_stage: dict[str, float] = {}
        self._cost_by_model: dict[str, float] = {}
        self._tokens_in: int = 0
        self._tokens_out: int = 0

    def record_usage(
        self,
        tokens_in: int,
        tokens_out: int,
        model: str,
        cost: float,
        stage: str = "",
    ) -> None:
        """Record token usage and cost. Raises if budget exceeded."""
        self._tokens_in += tokens_in
        self._tokens_out += tokens_out
        self.current_cost += cost
        self._cost_by_model[model] = self._cost_by_model.get(model, 0.0) + cost
        if stage:
            self._cost_by_stage[stage] = self._cost_by_stage.get(stage, 0.0) + cost
        self.check_budget(stage)

    def check_budget(self, stage: str = "") -> None:
        """Raise :class:`BudgetExceededError` if over limit."""
        if self.max_cost is not None and self.current_cost > self.max_cost:
            raise BudgetExceededError(self.current_cost, self.max_cost, stage)

    def remaining(self) -> float | None:
        """Remaining budget in USD, or ``None`` if unlimited."""
        if self.max_cost is None:
            return None
        return max(0.0, self.max_cost - self.current_cost)

    def estimate_stage(
        self, stage: str, item_count: int, model: str
    ) -> CostEstimate:
        """Rough cost estimate for a stage based on average tokens per item."""
        avg_in = 500
        avg_out = 800
        total_in = avg_in * item_count
        total_out = avg_out * item_count
        cost = estimate_cost(model, total_in, total_out)
        return CostEstimate(
            estimated_tokens=total_in + total_out,
            estimated_cost=cost,
            model=model,
            confidence="low",
        )

    def report(self) -> CostReport:
        """Return a snapshot of current cost state."""
        return CostReport(
            total_cost=self.current_cost,
            cost_by_stage=dict(self._cost_by_stage),
            cost_by_model=dict(self._cost_by_model),
            total_tokens_in=self._tokens_in,
            total_tokens_out=self._tokens_out,
        )
