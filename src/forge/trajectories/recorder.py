"""Trajectory recorder — captures agent actions into structured trajectories."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from forge.trajectories.models import Action, Trajectory, TrajectoryStep
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class TrajectoryRecorder:
    """Records agent actions step-by-step into a :class:`Trajectory`."""

    def __init__(self) -> None:
        self._current_id: str = ""
        self._task: str = ""
        self._steps: list[TrajectoryStep] = []

    def start(self, task: str) -> str:
        """Begin recording a new trajectory. Returns the trajectory ID."""
        self._current_id = str(uuid4())
        self._task = task
        self._steps = []
        logger.debug("Trajectory recording started: %s", self._current_id[:8])
        return self._current_id

    def record_step(
        self,
        observation: str,
        thought: str,
        action: Action,
        result: str,
    ) -> None:
        """Record a single step in the trajectory."""
        step = TrajectoryStep(
            observation=observation,
            thought=thought,
            action=action,
            result=result,
            timestamp=datetime.now(tz=timezone.utc),
        )
        self._steps.append(step)
        logger.debug(
            "Step %d recorded: tool=%s", len(self._steps), action.tool,
        )

    def finish(self, outcome: str) -> Trajectory:
        """Finalise the current trajectory and return it."""
        trajectory = Trajectory(
            trajectory_id=self._current_id,
            steps=list(self._steps),
            task=self._task,
            outcome=outcome,
        )
        logger.info(
            "Trajectory %s finished: %d steps, outcome=%s",
            self._current_id[:8], len(self._steps), outcome,
        )
        # Reset state
        self._current_id = ""
        self._task = ""
        self._steps = []
        return trajectory

    def export(self, trajectory: Trajectory, path: str) -> None:
        """Save a trajectory as a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(trajectory.model_dump(), f, indent=2, default=str)
        logger.info("Trajectory exported to %s", path)
