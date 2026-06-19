"""Trajectory formatter — converts trajectories into training-ready formats."""

from __future__ import annotations

import json
from pathlib import Path

from forge.trajectories.models import Trajectory
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class TrajectoryFormatter:
    """Converts :class:`Trajectory` instances into formats suitable for fine-tuning."""

    def to_react(self, trajectory: Trajectory) -> list[dict]:
        """Format as ReAct-style: observation → thought → action → result.

        Returns a list of message dicts suitable for chat fine-tuning.
        """
        messages: list[dict] = [
            {"role": "system", "content": f"You are an agent solving: {trajectory.task}"},
        ]
        for step in trajectory.steps:
            messages.append({
                "role": "user",
                "content": f"Observation: {step.observation}",
            })
            messages.append({
                "role": "assistant",
                "content": (
                    f"Thought: {step.thought}\n"
                    f"Action: {step.action.tool}({json.dumps(step.action.input, default=str)})\n"
                    f"Result: {step.result}"
                ),
            })
        return messages

    def to_chatml(self, trajectory: Trajectory) -> list[dict]:
        """Format as ChatML with tool_calls.

        Each step becomes a pair of assistant (with tool_call) and tool messages.
        """
        messages: list[dict] = [
            {"role": "system", "content": f"You are an agent solving: {trajectory.task}"},
        ]
        for step in trajectory.steps:
            # User observation
            messages.append({
                "role": "user",
                "content": step.observation,
            })
            # Assistant with tool call
            messages.append({
                "role": "assistant",
                "content": step.thought,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": step.action.tool,
                            "arguments": json.dumps(step.action.input, default=str),
                        },
                    }
                ],
            })
            # Tool result
            messages.append({
                "role": "tool",
                "name": step.action.tool,
                "content": step.result,
            })
        return messages

    def to_jsonl(self, trajectories: list[Trajectory], path: str) -> None:
        """Export multiple trajectories as JSONL (one trajectory per line)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for traj in trajectories:
                line = json.dumps(traj.model_dump(), default=str, ensure_ascii=False)
                f.write(line + "\n")
        logger.info("Exported %d trajectories to %s", len(trajectories), path)
