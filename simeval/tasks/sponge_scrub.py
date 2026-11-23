"""Pick up a sponge and scrub a stain off the counter.

Stain model: a grid of thin, visual-only cells on the counter, each with a dirt level in [0, 1]. A cell loses dirt in
proportion to how far the sponge's contact face slides over it while the sponge presses on the counter with at least
`min_press_force` (and, by default, only while it is held). Cells fade on camera as they clean, so a closed-loop policy
sees its own progress. This scores the *behaviour* we care about (pressure + motion over the stain), not mere contact.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..sim.contacts import is_grasped, total_normal_force
from ..sim.randomization import Sampler
from .base import Task


def _yaw_quat(deg: float) -> list[float]:
    a = np.radians(deg) / 2
    return [np.cos(a), 0.0, 0.0, np.sin(a)]


class SpongeScrubTask(Task):
    name = "sponge_scrub"
    stages = ["reached", "grasped", "lifted", "stain_contact", "half_clean", "clean"]

    def sample_task(self, s: Sampler) -> dict:
        c, sp, st = self.scene["counter"], self.scene["sponge"], self.scene["stain"]
        return {
            "counter": {"friction": s("counter.friction", c["friction"]), "rgba": s("counter.rgba", c["rgba"])},
            "sponge": {
                "scale": s("sponge.size_scale", sp["size_scale"]),
                "mass": s("sponge.mass", sp["mass"]),
                "friction": s("sponge.friction", sp["friction"]),
                "xy": s("sponge.start_xy", sp["start_xy"]),
                "yaw": s("sponge.start_yaw_deg", sp["start_yaw_deg"]),
            },
            "stain": {
                "center": s("stain.center_xy", st["center_xy"]),
                "radius": s("stain.radius", st["radius"]),
                "rgba": s("stain.rgba", st["rgba"]),
                "phases": s.rng.uniform(0, 2 * np.pi, 3).tolist(),
            },
        }

    # Scene ----------------------------------------------------------------------------------------------------------

    def build(self, spec: mujoco.MjSpec, v: dict) -> None:
        c, sp, st = self.scene["counter"], self.scene["sponge"], self.scene["stain"]
        G = mujoco.mjtGeom
        top = c["pos"][2] + c["half_size"][2]
        self.counter_top = top
        spec.worldbody.add_geom(name="counter", type=G.mjGEOM_BOX, pos=c["pos"], size=c["half_size"],
                                rgba=v["counter"]["rgba"], friction=[v["counter"]["friction"], 0.01, 0.001])

        hs = np.asarray(sp["half_size"]) * v["sponge"]["scale"]
        self.sponge_half = hs
        body = spec.worldbody.add_body(name="sponge", pos=[*v["sponge"]["xy"], top + hs[2] + 0.001],
                                       quat=_yaw_quat(v["sponge"]["yaw"]))
        body.add_freejoint(name="sponge_free")
        body.add_geom(name="sponge_collision", type=G.mjGEOM_BOX, size=hs, mass=v["sponge"]["mass"],
                      friction=[v["sponge"]["friction"], 0.01, 0.001], solref=sp["solref"],
                      solimp=[*sp["solimp"], 0.5, 2], rgba=sp["foam_rgba"])
        # Green scouring layer on the underside (visual only).
        body.add_geom(type=G.mjGEOM_BOX, size=[hs[0] * 0.999, hs[1] * 0.999, hs[2] * 0.3],
                      pos=[0, 0, -hs[2] * 0.72], rgba=sp["scour_rgba"], contype=0, conaffinity=0, mass=0)

        # Dirt map: cells inside a wobbly disk.
        cell = st["cell"]
        r0, cx, cy = v["stain"]["radius"], *v["stain"]["center"]
        ph = v["stain"]["phases"]
        rough = st["roughness"]
        n = int(np.ceil(1.5 * r0 / cell))
        centers = []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                x, y = i * cell, j * cell
                th = np.arctan2(y, x)
                rmax = r0 * (1 + rough * (np.sin(2 * th + ph[0]) + np.sin(3 * th + ph[1]) + np.sin(5 * th + ph[2])) / 3)
                if np.hypot(x, y) <= rmax:
                    centers.append((cx + x, cy + y))
        self.cell_centers = np.asarray(centers)
        self.stain_rgba = np.asarray(v["stain"]["rgba"], float)
        for k, (x, y) in enumerate(centers):
            spec.worldbody.add_geom(name=f"stain_{k}", type=G.mjGEOM_BOX, pos=[x, y, top + 0.0002],
                                    size=[cell / 2, cell / 2, 0.0002], rgba=self.stain_rgba,
                                    contype=0, conaffinity=0, group=2)

    # Scoring --------------------------------------------------------------------------------------------------------

    def reset(self, model, data, robot, variation) -> None:
        super().reset(model, data, robot, variation)
        self.sponge_id = model.body("sponge").id
        self.counter_body = {model.geom("counter").bodyid[0]}
        self.stain_geoms = np.array([model.geom(f"stain_{k}").id for k in range(len(self.cell_centers))])
        self.dirt = np.ones(len(self.cell_centers))
        self.prev_pose = None
        self.rest_z = None
        sc = self.scene["scoring"]
        self.extra = {"cells": int(len(self.dirt)), "cleaned_fraction": 0.0, "dirt_removed": 0.0,
                      "max_press_force": 0.0, "max_robot_counter_force": 0.0, "arm_arm_contact_steps": 0,
                      "sponge_dropped": False, "grasped_steps": 0, "scrub_steps": 0}
        self.sc = sc

    def _sponge_pose(self):
        d = self.data.body(self.sponge_id)
        return d.xpos.copy(), d.xmat.reshape(3, 3).copy()

    def step(self) -> None:
        m, d, sc = self.model, self.data, self.sc
        pos, rot = self._sponge_pose()
        if self.rest_z is None:
            self.rest_z = pos[2]

        # Reach and grasp, for any arm.
        sponge = {self.sponge_id}
        grasped = False
        for mount, arm in self.robot.arms.items():
            if np.linalg.norm(d.site(arm.tcp_site).xpos - pos) < sc["reach_radius"]:
                self.mark("reached")
            if is_grasped(m, d, self.robot.finger_ids[mount], sponge):
                grasped = True
        if grasped:
            self.mark("grasped")
            self.extra["grasped_steps"] += 1
            if pos[2] > self.rest_z + sc["lift_height"]:
                self.mark("lifted")

        # Scrubbing: pressure on the counter + sliding of the contact face over each cell.
        press = total_normal_force(m, d, sponge, self.counter_body)
        self.extra["max_press_force"] = max(self.extra["max_press_force"], press)
        if self.prev_pose is not None and press >= sc["min_press_force"] and (grasped or not sc["require_grasp"]):
            # Face pointing down: the sponge axis most aligned with world z; footprint spans the other two.
            down = int(np.argmax(np.abs(rot[2])))
            plane = [a for a in range(3) if a != down]
            rel = np.c_[self.cell_centers, np.full(len(self.cell_centers), self.counter_top)] - pos
            local = rel @ rot  # cell centres in the sponge frame
            inside = np.all(np.abs(local[:, plane]) <= self.sponge_half[plane], axis=1)
            if inside.any():
                ppos, prot = self.prev_pose
                prev_world = local[inside] @ prot.T + ppos  # where that face point was a step ago
                slide = np.linalg.norm((rel[inside] + pos - prev_world)[:, :2], axis=1)
                before = self.dirt[inside].copy()
                self.dirt[inside] = np.clip(before - slide / sc["scrub_distance_to_clean"], 0, 1)
                self.mark("stain_contact")
                self.extra["scrub_steps"] += 1
        self.prev_pose = (pos, rot)

        # Visual feedback: cleaned cells fade out.
        m.geom_rgba[self.stain_geoms, 3] = self.stain_rgba[3] * self.dirt

        cleaned = float(np.mean(self.dirt <= 1e-6))
        self.extra["cleaned_fraction"] = cleaned
        self.extra["dirt_removed"] = float(1 - self.dirt.mean())
        if cleaned >= 0.5:
            self.mark("half_clean")
        if cleaned >= sc["success_fraction"]:
            self.mark("clean")

        # Safety: robot pressing on the counter, and the two arms touching each other.
        robot_bodies = self.robot.body_ids
        self.extra["max_robot_counter_force"] = max(
            self.extra["max_robot_counter_force"], total_normal_force(m, d, robot_bodies, self.counter_body))
        mounts = list(self.robot.arms)
        if len(mounts) == 2:
            a = {i for i in robot_bodies if m.body(i).name.startswith(mounts[0] + "_")}
            b = robot_bodies - a
            if total_normal_force(m, d, a, b) > 0:
                self.extra["arm_arm_contact_steps"] += 1
        if pos[2] < self.counter_top - 0.05:
            self.extra["sponge_dropped"] = True

    def success(self) -> bool:
        return self.extra.get("cleaned_fraction", 0.0) >= self.sc["success_fraction"]

    def failed(self) -> bool:
        return self.extra.get("sponge_dropped", False)
