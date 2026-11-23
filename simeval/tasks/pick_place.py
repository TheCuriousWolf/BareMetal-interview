"""Pick an object from a start pose and place it in a target region. Start and target come from the scene file."""

from __future__ import annotations

import mujoco
import numpy as np

from ..sim.contacts import is_grasped
from ..sim.randomization import Sampler
from .base import Task
from .sponge_scrub import _yaw_quat

SHAPES = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
          "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}


class PickPlaceTask(Task):
    name = "pick_place"
    stages = ["reached", "grasped", "lifted", "over_target", "placed"]

    def sample_task(self, s: Sampler) -> dict:
        t, o, g = self.scene["table"], self.scene["object"], self.scene["target"]
        return {
            "table": {"friction": s("table.friction", t["friction"])},
            "object": {"mass": s("object.mass", o["mass"]), "friction": s("object.friction", o["friction"]),
                       "xy": s("object.start_xy", o["start_xy"]), "yaw": s("object.start_yaw_deg", o["start_yaw_deg"])},
            "target": {"center": s("target.center_xy", g["center_xy"])},
        }

    def build(self, spec: mujoco.MjSpec, v: dict) -> None:
        t, o, g = self.scene["table"], self.scene["object"], self.scene["target"]
        G = mujoco.mjtGeom
        self.top = t["pos"][2] + t["half_size"][2]
        spec.worldbody.add_geom(name="table", type=G.mjGEOM_BOX, pos=t["pos"], size=t["half_size"], rgba=t["rgba"],
                                friction=[v["table"]["friction"], 0.01, 0.001])
        hs = np.asarray(o["half_size"], float)
        height = hs[2] if o["shape"] != "sphere" else hs[0]
        body = spec.worldbody.add_body(name="object", pos=[*v["object"]["xy"], self.top + height + 0.001],
                                       quat=_yaw_quat(v["object"]["yaw"]))
        body.add_freejoint()
        body.add_geom(type=SHAPES[o["shape"]], size=hs, mass=v["object"]["mass"], rgba=o["rgba"],
                      friction=[v["object"]["friction"], 0.01, 0.001])
        spec.worldbody.add_geom(name="target_marker", type=G.mjGEOM_CYLINDER, pos=[*v["target"]["center"], self.top],
                                size=[g["radius"], 0.0003, 0], rgba=g["rgba"], contype=0, conaffinity=0)

    def reset(self, model, data, robot, variation) -> None:
        super().reset(model, data, robot, variation)
        self.obj = model.body("object").id
        self.target = np.asarray(variation["target"]["center"], float)
        self.sc = self.scene["scoring"]
        self.rest_z = None
        self.extra = {"final_distance": None, "dropped": False}

    def step(self) -> None:
        m, d, sc = self.model, self.data, self.sc
        pos = d.body(self.obj).xpos
        self.rest_z = pos[2] if self.rest_z is None else self.rest_z
        held = False
        for mount, arm in self.robot.arms.items():
            if np.linalg.norm(d.site(arm.tcp_site).xpos - pos) < sc["reach_radius"]:
                self.mark("reached")
            held |= is_grasped(m, d, self.robot.finger_ids[mount], {self.obj})
        if held:
            self.mark("grasped")
            if pos[2] > self.rest_z + sc["lift_height"]:
                self.mark("lifted")
        dist = float(np.linalg.norm(pos[:2] - self.target))
        self.extra["final_distance"] = dist
        if dist < self.scene["target"]["radius"]:
            self.mark("over_target")
            speed = np.linalg.norm(d.qvel[m.body_dofadr[self.obj]: m.body_dofadr[self.obj] + 3])
            if not held and speed < sc["settle_speed"] and pos[2] < self.rest_z + 0.01:
                self.mark("placed")
        self.extra["dropped"] = bool(pos[2] < self.top - 0.05)

    def success(self) -> bool:
        return self.stage_times.get("placed") is not None

    def failed(self) -> bool:
        return self.extra["dropped"]
