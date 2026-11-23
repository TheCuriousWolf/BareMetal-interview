"""Embodiment-neutral data passed between the simulator, the runner and a policy.

Everything a policy sees or returns goes through these types, so a policy adapter never touches MuJoCo and a task
never touches a model provider's wire format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Observation:
    """One observation of the robot and scene.

    images: camera name -> HxWx3 uint8 RGB, at the robot's native camera resolution.
    state:  arm name -> joint positions followed by gripper opening (0 closed .. 1 open).
    """

    images: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    instruction: str
    time: float
    sensors: dict[str, np.ndarray] = field(default_factory=dict)  # joint velocities, efforts, end-effector poses
    last_action: dict[str, np.ndarray] | None = None


@dataclass
class ActionChunk:
    """A sequence of absolute joint targets per arm, executed one row per control step.

    actions: arm name -> (T, dof) array, same layout as Observation.state.
    """

    actions: dict[str, np.ndarray]
    info: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return min(len(a) for a in self.actions.values()) if self.actions else 0

    def row(self, i: int) -> dict[str, np.ndarray]:
        return {arm: a[i] for arm, a in self.actions.items()}


@dataclass
class PolicySpec:
    """What a policy needs from the runner.

    obs_period_s:  how often the policy wants a rendered observation (rendering is the costly part of a step).
    warmup_s:      simulated time to hold still before the first query, e.g. to fill a frame history.
    max_chunk_rows: rows of each chunk to execute before asking again (open-loop horizon).
    """

    name: str
    obs_period_s: float = 0.2
    warmup_s: float = 0.0
    max_chunk_rows: int = 24
