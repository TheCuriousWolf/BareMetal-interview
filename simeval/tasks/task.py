"""A task defined entirely in YAML: a scene (interactables) plus an evaluation (metrics, stages, end conditions).

    scene: counter_sponge.yaml          # scenes/: what is in the world and where
    instruction: "pick up the sponge and scrub the stain off the counter"
    max_seconds: 90
    metrics:                            # measured every step (see simeval/tasks/metrics.py)
      eef_to_sponge: {type: distance, a: robot.any.eef, b: sponge}
      sponge_held:   {type: grasped, object: sponge}
      stain_cleaned: {type: value, object: stain, key: cleaned}
    stages:                             # checks in order of the task; the first time each holds is recorded
      - {name: reached, when: eef_to_sponge < 0.06}
      - {name: grasped, when: sponge_held > 0, hold_for: 0.25}
    end:
      success: stain_cleaned >= 0.8     # ends the episode as a success
      failure: [sponge_height < -0.05]  # any of these ends it as a failure

The same class runs every task; a new task is a new YAML file (and, for new kinds of objects, a new Interactable).
"""

from __future__ import annotations

import copy

import mujoco
import yaml

from ..paths import SCENES
from ..scene.interactables import TYPES as OBJECT_TYPES
from ..sim.randomization import Sampler
from . import metrics as M


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def as_list(x) -> list:
    return [] if x is None else x if isinstance(x, list) else [x]


class Task:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg.get("name", cfg.get("task", "task"))
        scene = yaml.safe_load((SCENES / cfg["scene"]).read_text())
        self.scene = deep_merge(scene, cfg.get("scene_overrides", {}))
        self.instruction = cfg.get("instruction", "")
        self.max_seconds = float(cfg.get("max_seconds", 60))
        self.stage_defs = cfg.get("stages", [])
        self.stages = [s["name"] for s in self.stage_defs]
        end = cfg.get("end", {})
        self.success_when = as_list(end.get("success"))
        self.failure_when = as_list(end.get("failure"))

    # Scene ----------------------------------------------------------------------------------------------------------
    def sample_variation(self, s: Sampler) -> dict:
        common = self.scene.get("common", {})
        self.objects = {}
        for name, params in self.scene["objects"].items():
            self.objects[name] = OBJECT_TYPES[params["type"]](name, params, self.objects)
        return {
            "lighting": {"intensity": s("lighting.intensity", common.get("light_intensity", 1.0)),
                         "dir_jitter": s("lighting.dir_jitter", common.get("light_dir_jitter", [0, 0, 0]))},
            "camera_jitter": {
                name: {"pos": s(f"camera.{name}.pos", c.get("pos", [0, 0, 0])),
                       "rot_deg": s(f"camera.{name}.rot_deg", c.get("rot_deg", [0, 0, 0]))}
                for name, c in common.get("camera_jitter", {}).items()},
            "objects": {name: obj.sample(s) for name, obj in self.objects.items()},
        }

    def build(self, spec: mujoco.MjSpec, variation: dict) -> None:
        for name, obj in self.objects.items():
            obj.build(spec, variation["objects"][name])

    # Episode --------------------------------------------------------------------------------------------------------
    def reset(self, model, data, robot, variation) -> None:
        self.model, self.data = model, data
        for obj in self.objects.values():
            obj.bind(model, data, robot)
        world = M.World(model, data, robot, self.objects)
        self.metrics_now = {n: M.TYPES[c["type"]](n, c, world) for n, c in self.cfg.get("metrics", {}).items()}
        self.stage_times = {s: None for s in self.stages}
        self.held_since: dict[str, float] = {}
        self.last_time = float(data.time)
        self._success = self._failure = False

    def step(self) -> None:
        t = float(self.data.time)
        dt, self.last_time = max(0.0, t - self.last_time), t
        for obj in self.objects.values():
            obj.step()
        for m in self.metrics_now.values():
            m.update(dt)
        for st in self.stage_defs:
            name = st["name"]
            if self.stage_times[name] is not None:
                continue
            if M.check(st["when"], self.metrics_now):
                self.held_since.setdefault(name, t)
                if t - self.held_since[name] >= st.get("hold_for", 0.0):
                    self.stage_times[name] = self.held_since[name]
            else:
                self.held_since.pop(name, None)
        self._success = bool(self.success_when) and all(M.check(c, self.metrics_now) for c in self.success_when)
        self._failure = any(M.check(c, self.metrics_now) for c in self.failure_when)

    def success(self) -> bool:
        return self._success

    def failed(self) -> bool:
        return self._failure

    def metrics(self) -> dict:
        out = {
            "success": self._success,
            "failure": self._failure,
            "progress": sum(v is not None for v in self.stage_times.values()) / max(1, len(self.stages)),
            "stage_times": dict(self.stage_times),
        }
        for name, m in self.metrics_now.items():
            r = m.result()
            out[name] = r["final"]
            out[f"{name}.max"] = r["max"]
            out[f"{name}.min"] = r["min"]
        return out

