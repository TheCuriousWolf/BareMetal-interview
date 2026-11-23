"""SimEnv: one robot config + one task, stepped at the control rate.

Policies act at `control_hz` (20 Hz, the YAM station rate); physics runs at 1/timestep (500 Hz). Each control step
holds the commanded joint targets for `physics_steps` substeps, like the real position servos between commands.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..core.types import Observation
from .cameras import Renderer
from .randomization import DEFAULT_LEVELS, Sampler
from .world import compose


class SimEnv:
    def __init__(self, robot_cfg: dict, task, control_hz: float = 20.0, levels: dict | None = None):
        self.robot_cfg = robot_cfg
        self.task = task
        self.control_hz = control_hz
        self.levels = levels or DEFAULT_LEVELS
        self.renderer = None

    def reset(self, seed: int, level: str = "none") -> dict:
        """Build a fresh scene for this episode. Returns the sampled variation (stored with the results)."""
        self.close()
        sampler = Sampler(np.random.default_rng(seed), self.levels[level])
        variation = self.task.sample_variation(sampler)
        self.variation = variation
        self.model, self.robot = compose(self.robot_cfg, self.task, variation)
        self.data = mujoco.MjData(self.model)
        self.robot.reset(self.data)
        mujoco.mj_forward(self.model, self.data)
        self.task.reset(self.model, self.data, self.robot, variation)
        self.physics_steps = max(1, round(1.0 / (self.control_hz * self.model.opt.timestep)))
        self.renderer = Renderer(self.model, self.robot.cameras)
        # Let objects settle onto the counter before the clock that scoring sees starts.
        for _ in range(10):
            self.hold()
        self.data.time = 0.0
        return {"seed": seed, "level": level, "params": sampler.record}

    @property
    def time(self) -> float:
        return float(self.data.time)

    def state(self) -> dict[str, np.ndarray]:
        return self.robot.state(self.data)

    def step(self, targets: dict[str, np.ndarray]) -> dict[str, float]:
        clipped = self.robot.apply(self.data, targets)
        mujoco.mj_step(self.model, self.data, nstep=self.physics_steps)
        self.task.step()
        return clipped

    def hold(self) -> None:
        """Advance one control step keeping the current actuator targets."""
        mujoco.mj_step(self.model, self.data, nstep=self.physics_steps)
        self.task.step()

    def observe(self, render: bool = True) -> Observation:
        images = self.renderer.render(self.data) if render else {}
        return Observation(images=images, state=self.state(), instruction=self.task.instruction, time=self.time,
                           sensors=self.robot.sensors(self.data), last_action=self.robot.last_action)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
