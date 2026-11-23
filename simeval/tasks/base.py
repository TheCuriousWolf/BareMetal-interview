"""Task interface.

A task owns three things:
  1. its scene file (scenes/<name>.yaml): fixtures, objects and start/target parameters, all overridable from config;
  2. scene construction (`build`) on top of the shared world;
  3. scoring from ground-truth simulator state (`step`, `metrics`), never from anything the policy reports.

Scoring is staged: each stage records the first sim time it was reached, so a policy that grasps but never cleans
still gets measurable partial progress, which is what separates zero-shot models that all fail the full task.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import mujoco
import numpy as np
import yaml

from ..paths import SCENES
from ..sim.randomization import Sampler


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Task(ABC):
    name = "task"
    stages: list[str] = []

    def __init__(self, cfg: dict):
        self.cfg = cfg
        scene_file = SCENES / cfg["scene"] if "scene" in cfg else None
        scene = yaml.safe_load(scene_file.read_text()) if scene_file else {}
        self.scene = deep_merge(scene, cfg.get("scene_overrides", {}))
        self.instruction = cfg.get("instruction", "")
        self.max_seconds = float(cfg.get("max_seconds", 60))

    # Episode lifecycle --------------------------------------------------------------------------------------------

    def sample_variation(self, sampler: Sampler) -> dict:
        """Draw every randomized quantity for one episode. Common groups first, task-specific in `sample_task`."""
        common = self.scene.get("common", {})
        variation = {
            "lighting": {
                "intensity": sampler("lighting.intensity", common.get("light_intensity", 1.0)),
                "dir_jitter": sampler("lighting.dir_jitter", common.get("light_dir_jitter", [0, 0, 0])),
            },
            "camera_jitter": {
                name: {"pos": sampler(f"camera.{name}.pos", c.get("pos", [0, 0, 0])),
                       "rot_deg": sampler(f"camera.{name}.rot_deg", c.get("rot_deg", [0, 0, 0]))}
                for name, c in common.get("camera_jitter", {}).items()
            },
        }
        variation.update(self.sample_task(sampler))
        return variation

    @abstractmethod
    def sample_task(self, sampler: Sampler) -> dict: ...

    @abstractmethod
    def build(self, spec: mujoco.MjSpec, variation: dict) -> None:
        """Add fixtures and objects to the scene spec."""

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, robot, variation: dict) -> None:
        """Called after the robot is at home and before the first step."""
        self.model, self.data, self.robot, self.variation = model, data, robot, variation
        self.stage_times: dict[str, float | None] = {s: None for s in self.stages}
        self.extra: dict = {}

    @abstractmethod
    def step(self) -> None:
        """Update progress after each control step."""

    def mark(self, stage: str) -> None:
        if self.stage_times.get(stage) is None:
            self.stage_times[stage] = float(self.data.time)

    @abstractmethod
    def success(self) -> bool: ...

    def failed(self) -> bool:
        """Unrecoverable state (e.g. object dropped off the counter): ends the episode early."""
        return False

    def metrics(self) -> dict:
        reached = [s for s in self.stages if self.stage_times[s] is not None]
        return {
            "success": bool(self.success()),
            "progress": len(reached) / max(1, len(self.stages)),
            "stage_times": dict(self.stage_times),
            **self.extra,
        }


def body_pos(data, name) -> np.ndarray:
    return data.body(name).xpos.copy()
