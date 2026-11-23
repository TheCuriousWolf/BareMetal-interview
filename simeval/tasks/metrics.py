"""Metric types for the evaluation setup. Each one is measured every control step from simulator state.

Things a metric can point at:
    robot.any.eef        end effector (TCP) of whichever arm is closest / any arm
    robot.left.eef       end effector of one mount
    robot.any.gripper    finger bodies of any arm          robot.left.gripper   ... of one mount
    robot.all            every robot body
    sponge               an interactable (its bodies, or its centre for distances)
    sponge.top           a marker on an interactable

Metric types (`type:` in config) and their parameters:
    distance       a, b                 metres between the closest pair of points
    contact        a, b, [min_force]    1 while touching, else 0
    contact_time   a, b, [min_force]    seconds of contact so far
    force          a, b                 total normal force (N) between them
    grasped        object, [arm]        1 while the gripper holds the object
    grasp_time     object, [arm]        seconds held so far
    height         object               rise above its start height (m)
    moved          object               horizontal distance from its start (m)
    in_region      object, region       1 while the object's centre is inside the region
    value          object, key          a value the interactable reports, e.g. a spill's `cleaned`
Every metric keeps its current value plus its max and min over the episode.
"""

from __future__ import annotations

import numpy as np

from ..sim.contacts import is_grasped, total_normal_force


class Metric:
    def __init__(self, name: str, cfg: dict, world):
        self.name, self.cfg, self.world = name, cfg, world
        self.value = 0.0
        self.max = -np.inf
        self.min = np.inf

    def update(self, dt: float) -> None:
        self.value = float(self.measure(dt))
        self.max = max(self.max, self.value)
        self.min = min(self.min, self.value)

    def measure(self, dt: float) -> float:
        raise NotImplementedError

    def result(self) -> dict:
        return {"final": self.value, "max": self.max, "min": self.min}


class World:
    """Resolves the names used in metric configs to points and body sets."""

    def __init__(self, model, data, robot, objects):
        self.model, self.data, self.robot, self.objects = model, data, robot, objects

    def _mounts(self, which: str) -> list[str]:
        return list(self.robot.arms) if which == "any" else [which]

    def points(self, ref: str) -> list[np.ndarray]:
        parts = ref.split(".")
        if parts[0] == "robot":
            if parts[-1] == "eef":
                return [self.data.site(self.robot.arms[m].tcp_site).xpos.copy() for m in self._mounts(parts[1])]
            return [self.data.xpos[i].copy() for i in self.bodies(ref)]
        obj = self.objects[parts[0]]
        return [obj.position(parts[1] if len(parts) > 1 else None)]

    def bodies(self, ref: str) -> set[int]:
        parts = ref.split(".")
        if parts[0] == "robot":
            if parts[1] == "all":
                return set(self.robot.body_ids)
            ids = set()
            for m in self._mounts(parts[1]):
                ids |= self.robot.finger_ids[m] if parts[-1] == "gripper" else \
                    {i for i in self.robot.body_ids if self.model.body(i).name.startswith(m + "_")}
            return ids
        return self.objects[parts[0]].body_ids


class Distance(Metric):
    def measure(self, dt):
        a, b = self.world.points(self.cfg["a"]), self.world.points(self.cfg["b"])
        return min(np.linalg.norm(p - q) for p in a for q in b)


class Force(Metric):
    def measure(self, dt):
        w = self.world
        return total_normal_force(w.model, w.data, w.bodies(self.cfg["a"]), w.bodies(self.cfg["b"]))


class Contact(Force):
    def measure(self, dt):
        return float(super().measure(dt) > self.cfg.get("min_force", 0.0) or self._touching())

    def _touching(self):
        if self.cfg.get("min_force", 0.0) > 0:
            return False
        w, a, b = self.world, self.world.bodies(self.cfg["a"]), self.world.bodies(self.cfg["b"])
        for i in range(w.data.ncon):
            c = w.data.contact[i]
            b1, b2 = w.model.geom_bodyid[c.geom1], w.model.geom_bodyid[c.geom2]
            if (b1 in a and b2 in b) or (b1 in b and b2 in a):
                return True
        return False


class ContactTime(Contact):
    total = 0.0

    def measure(self, dt):
        self.total += dt * super().measure(dt)
        return self.total


class Grasped(Metric):
    def measure(self, dt):
        w = self.world
        mounts = w._mounts(self.cfg.get("arm", "any"))
        obj = w.bodies(self.cfg["object"])
        return float(any(is_grasped(w.model, w.data, w.robot.finger_ids[m], obj) for m in mounts))


class GraspTime(Grasped):
    total = 0.0

    def measure(self, dt):
        self.total += dt * super().measure(dt)
        return self.total


class Height(Metric):
    def measure(self, dt):
        obj = self.world.objects[self.cfg["object"]]
        return obj.position()[2] - obj.start_pos[2]


class Moved(Metric):
    def measure(self, dt):
        obj = self.world.objects[self.cfg["object"]]
        return float(np.linalg.norm(obj.position()[:2] - obj.start_pos[:2]))


class InRegion(Metric):
    def measure(self, dt):
        obj = self.world.objects[self.cfg["object"]]
        return float(self.world.objects[self.cfg["region"]].contains(obj.position()))


class Value(Metric):
    def measure(self, dt):
        return self.world.objects[self.cfg["object"]].values()[self.cfg["key"]]


TYPES = {"distance": Distance, "contact": Contact, "contact_time": ContactTime, "force": Force,
         "grasped": Grasped, "grasp_time": GraspTime, "height": Height, "moved": Moved,
         "in_region": InRegion, "value": Value}


# Conditions -----------------------------------------------------------------------------------------------------------

OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.isclose}


def check(condition: str, metrics: dict[str, Metric]) -> bool:
    """Evaluate `"<metric> <op> <number>"`, e.g. `"stain_cleaned >= 0.8"`. Several joined by ` and ` all must hold."""
    for part in condition.split(" and "):
        name, op, number = part.split()
        if not OPS[op](metrics[name].value, float(number)):
            return False
    return True
