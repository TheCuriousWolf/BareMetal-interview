"""Reference policies that need no model: they bound the metric from below and sanity-check the task."""

from __future__ import annotations

import numpy as np

from ..core.types import ActionChunk, Observation
from .base import Policy


class HoldPolicy(Policy):
    """Keeps every arm where it is. Any score above this one is caused by the policy, not by the scene."""

    def act(self, obs: Observation) -> ActionChunk:
        n = self.spec.max_chunk_rows
        return ActionChunk({arm: np.repeat(q[None], n, axis=0) for arm, q in obs.state.items()})


class RandomPolicy(Policy):
    """Small random joint perturbations around the current state (a noise floor for contact-based metrics)."""

    def reset(self, instruction, robot_info):
        super().reset(instruction, robot_info)
        self.rng = np.random.default_rng(self.cfg.get("seed", 0))

    def act(self, obs: Observation) -> ActionChunk:
        n, scale = self.spec.max_chunk_rows, self.cfg.get("scale", 0.05)
        out = {}
        for arm, q in obs.state.items():
            steps = np.cumsum(self.rng.normal(0, scale / np.sqrt(n), (n, len(q))), axis=0)
            rows = q[None] + steps
            rows[:, -1] = np.clip(rows[:, -1], 0, 1)
            out[arm] = rows
        return ActionChunk(out)
