"""Interactables: everything a scene is built from.

Each interactable is declared in a scene file under `objects:` with a `type` and its parameters:

    objects:
      counter: {type: surface, pos: [0, -0.12, 0.73], half_size: [0.65, 0.55, 0.03]}
      sponge:  {type: sponge, surface: counter, xy: [-0.13, 0.07]}
      stain:   {type: spill, surface: counter, center_xy: [0.06, -0.02], radius: 0.04, erased_by: sponge}

Every interactable
  * adds itself to the MuJoCo scene (`build`),
  * exposes its bodies (for contact metrics) and named markers (points used by distance metrics),
  * may update itself every step (`step`, e.g. a spill being wiped) and report values (`values`, e.g. `cleaned`).

To add a new kind of thing (towel, toothpaste cap), subclass Interactable and register it in TYPES.
Any parameter can be fixed or randomized ({nominal, range | jitter | choices, group}); see simeval/sim/randomization.py.
"""

from __future__ import annotations

import mujoco
import numpy as np

G = mujoco.mjtGeom


def yaw_quat(deg: float) -> list[float]:
    a = np.radians(deg) / 2
    return [float(np.cos(a)), 0.0, 0.0, float(np.sin(a))]


class Interactable:
    type = "interactable"

    def __init__(self, name: str, params: dict, scene: "dict[str, Interactable]"):
        self.name = name
        self.params = params
        self.scene = scene  # all interactables by name, for references such as `surface: counter`
        self.markers: dict[str, list[float]] = dict(params.get("markers", {}))  # extra user markers, local frame

    # Build ----------------------------------------------------------------------------------------------------------
    def sample(self, s) -> dict:
        """Draw this object's randomized parameters. `s(key, value)` samples one value."""
        return {}

    def build(self, spec: mujoco.MjSpec, v: dict) -> None:
        raise NotImplementedError

    def add_markers(self, body, defaults: dict[str, list[float]]) -> None:
        for marker, pos in {**defaults, **self.markers}.items():
            body.add_site(name=f"{self.name}.{marker}", pos=pos, size=[0.004, 0, 0], rgba=[1, 0, 1, 0])

    # Run ------------------------------------------------------------------------------------------------------------
    def bind(self, model: mujoco.MjModel, data: mujoco.MjData, robot) -> None:
        self.model, self.data, self.robot = model, data, robot
        self.body_ids = {i for i in range(model.nbody) if model.body(i).name == self.name}
        self.start_pos = self.position()

    def step(self) -> None:
        pass

    def position(self, marker: str | None = None) -> np.ndarray:
        if marker:
            return self.data.site(f"{self.name}.{marker}").xpos.copy()
        return self.data.body(self.name).xpos.copy() if self.body_ids else np.zeros(3)

    def values(self) -> dict[str, float]:
        return {}


class Surface(Interactable):
    """Static box: counter, table, shelf."""

    type = "surface"

    def sample(self, s):
        p = self.params
        return {"friction": s(f"{self.name}.friction", p.get("friction", 1.0)),
                "rgba": s(f"{self.name}.rgba", p.get("rgba", [0.72, 0.70, 0.66, 1]))}

    @property
    def top(self) -> float:
        return self.params["pos"][2] + self.params["half_size"][2]

    def build(self, spec, v):
        body = spec.worldbody.add_body(name=self.name, pos=self.params["pos"])
        body.add_geom(name=self.name, type=G.mjGEOM_BOX, size=self.params["half_size"], rgba=v["rgba"],
                      friction=[v["friction"], 0.01, 0.001])
        self.add_markers(body, {"center": [0, 0, self.params["half_size"][2]]})


class RigidObject(Interactable):
    """Free-moving object (box, cylinder, sphere) resting on a `surface`."""

    type = "object"
    SHAPES = {"box": G.mjGEOM_BOX, "cylinder": G.mjGEOM_CYLINDER, "sphere": G.mjGEOM_SPHERE}

    def sample(self, s):
        p, n = self.params, self.name
        return {"xy": s(f"{n}.xy", p["xy"]), "yaw": s(f"{n}.yaw_deg", p.get("yaw_deg", 0.0)),
                "scale": s(f"{n}.size_scale", p.get("size_scale", 1.0)), "mass": s(f"{n}.mass", p.get("mass", 0.1)),
                "friction": s(f"{n}.friction", p.get("friction", 1.0))}

    def half_height(self, size) -> float:
        return size[0] if self.params.get("shape", "box") == "sphere" else size[-1] if len(size) < 3 else size[2]

    def build(self, spec, v):
        p = self.params
        size = np.asarray(p["half_size"], float) * v["scale"]
        self.size = size
        top = self.scene[p["surface"]].top
        body = spec.worldbody.add_body(name=self.name, pos=[*v["xy"], top + self.half_height(size) + 0.001],
                                       quat=yaw_quat(v["yaw"]))
        body.add_freejoint(name=f"{self.name}_free")
        body.add_geom(name=self.name, type=self.SHAPES[p.get("shape", "box")], size=size, mass=v["mass"],
                      friction=[v["friction"], 0.01, 0.001], rgba=p.get("rgba", [0.85, 0.2, 0.2, 1]),
                      **self.contact_params())
        self.decorate(body, size)
        self.add_markers(body, {"center": [0, 0, 0], "top": [0, 0, self.half_height(size)],
                                "bottom": [0, 0, -self.half_height(size)]})

    def contact_params(self) -> dict:
        return {}

    def decorate(self, body, size) -> None:
        pass


class Sponge(RigidObject):
    """Kitchen sponge: rigid box with compliant contact (stands in for foam) and a green scouring layer."""

    type = "sponge"

    def contact_params(self):
        p = self.params
        return {"solref": p.get("solref", [0.02, 1.0]), "solimp": [*p.get("solimp", [0.7, 0.95, 0.006]), 0.5, 2]}

    def decorate(self, body, size):
        body.add_geom(type=G.mjGEOM_BOX, size=[size[0] * 0.999, size[1] * 0.999, size[2] * 0.3],
                      pos=[0, 0, -size[2] * 0.72], rgba=self.params.get("scour_rgba", [0.18, 0.45, 0.22, 1]),
                      contype=0, conaffinity=0, mass=0)


class Region(Interactable):
    """Flat target zone drawn on a surface (no physics). `in_region` metrics test objects against it."""

    type = "region"

    def sample(self, s):
        return {"center": s(f"{self.name}.center_xy", self.params["center_xy"])}

    def build(self, spec, v):
        top = self.scene[self.params["surface"]].top
        self.center = np.asarray(v["center"], float)
        self.radius = float(self.params.get("radius", 0.05))
        body = spec.worldbody.add_body(name=self.name, pos=[*self.center, top])
        body.add_geom(type=G.mjGEOM_CYLINDER, size=[self.radius, 0.0003, 0], contype=0, conaffinity=0,
                      rgba=self.params.get("rgba", [0.2, 0.6, 0.9, 0.5]))
        self.add_markers(body, {"center": [0, 0, 0]})

    def contains(self, point: np.ndarray) -> bool:
        return float(np.linalg.norm(point[:2] - self.center)) <= self.radius


class Spill(Interactable):
    """A stain or spill: a patch of decals on a surface, erased where a tool presses and slides over it.

    A decal loses dirt in proportion to how far the tool's contact face slides over it while the tool presses the
    surface with at least `min_press_force` (and, if `require_grasp`, only while the robot holds the tool).
    Erased decals fade on camera. Reports `cleaned` (fraction of decals fully gone) and `remaining` (mean dirt).
    """

    type = "spill"

    def sample(self, s):
        p, n = self.params, self.name
        return {"center": s(f"{n}.center_xy", p["center_xy"]), "radius": s(f"{n}.radius", p.get("radius", 0.04)),
                "rgba": s(f"{n}.rgba", p.get("rgba", [0.36, 0.22, 0.10, 0.9])),
                "phases": s.rng.uniform(0, 2 * np.pi, 3).tolist()}

    def build(self, spec, v):
        p = self.params
        cell, r0, (cx, cy), ph = p.get("cell", 0.008), v["radius"], v["center"], v["phases"]
        rough = p.get("roughness", 0.25)
        self.top = self.scene[p["surface"]].top
        n = int(np.ceil(1.5 * r0 / cell))
        cells = []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                x, y = i * cell, j * cell
                th = np.arctan2(y, x)
                rmax = r0 * (1 + rough * (np.sin(2 * th + ph[0]) + np.sin(3 * th + ph[1]) + np.sin(5 * th + ph[2])) / 3)
                if np.hypot(x, y) <= rmax:
                    cells.append((cx + x, cy + y))
        self.cells = np.asarray(cells)
        self.rgba = np.asarray(v["rgba"], float)
        body = spec.worldbody.add_body(name=self.name, pos=[cx, cy, self.top])
        for k, (x, y) in enumerate(cells):
            body.add_geom(name=f"{self.name}_{k}", type=G.mjGEOM_BOX, pos=[x - cx, y - cy, 0.0002],
                          size=[cell / 2, cell / 2, 0.0002], rgba=self.rgba, contype=0, conaffinity=0, group=2)
        self.add_markers(body, {"center": [0, 0, 0]})

    def bind(self, model, data, robot):
        super().bind(model, data, robot)
        self.geoms = np.array([model.geom(f"{self.name}_{k}").id for k in range(len(self.cells))])
        self.dirt = np.ones(len(self.cells))
        self.tool = self.scene[self.params["erased_by"]]
        self.surface = self.scene[self.params["surface"]]
        self.prev = None

    def step(self):
        from ..sim.contacts import is_grasped, total_normal_force

        p, m, d = self.params, self.model, self.data
        tool = d.body(self.tool.name)
        pos, rot = tool.xpos.copy(), tool.xmat.reshape(3, 3).copy()
        press = total_normal_force(m, d, self.tool.body_ids, self.surface.body_ids)
        held = any(is_grasped(m, d, f, self.tool.body_ids) for f in self.robot.finger_ids.values())
        if self.prev is not None and press >= p.get("min_press_force", 0.5) and (held or not p.get("require_grasp", True)):
            down = int(np.argmax(np.abs(rot[2])))  # the tool face pointing down
            plane = [a for a in range(3) if a != down]
            rel = np.c_[self.cells, np.full(len(self.cells), self.top)] - pos
            local = rel @ rot
            inside = np.all(np.abs(local[:, plane]) <= self.tool.size[plane], axis=1)
            if inside.any():
                ppos, prot = self.prev
                slide = np.linalg.norm((rel[inside] + pos - (local[inside] @ prot.T + ppos))[:, :2], axis=1)
                self.dirt[inside] = np.clip(self.dirt[inside] - slide / p.get("scrub_distance", 0.08), 0, 1)
        self.prev = (pos, rot)
        m.geom_rgba[self.geoms, 3] = self.rgba[3] * self.dirt

    def values(self):
        return {"cleaned": float(np.mean(self.dirt <= 1e-6)), "remaining": float(self.dirt.mean())}


TYPES = {c.type: c for c in (Surface, RigidObject, Sponge, Region, Spill)}
