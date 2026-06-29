"""Data models for agent trajectories."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Action(BaseModel):
    """A single agent action (tool call)."""

    tool: str  # "browser", "terminal", "api", etc.
    input: dict[str, Any] = Field(default_factory=dict)
    raw_command: str | None = None


class TrajectoryStep(BaseModel):
    """One step in an agent trajectory: observe → think → act → result."""

    observation: str
    thought: str
    action: Action
    result: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class Trajectory(BaseModel):
    """A complete agent trajectory for a single task."""

    trajectory_id: str = Field(default_factory=lambda: str(uuid4()))
    steps: list[TrajectoryStep] = Field(default_factory=list)
    task: str
    outcome: str = ""  # "success" | "failure" | "partial"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
