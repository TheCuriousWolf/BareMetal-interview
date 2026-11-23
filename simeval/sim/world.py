"""Scene composition: base world + robot + station cameras + task fixtures and objects, compiled per episode.

Rebuilding the model per episode (≈0.3 s) is what allows physical parameters (sizes, masses, friction, lighting,
camera pose) to be randomized, not only initial poses.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np

from ..paths import ROOT
from .cameras import CameraSpec, add_calibrated_camera
from ..registry import resolve
from .robot import spawn_robot


def base_spec(physics: dict | None = None, lighting: dict | None = None) -> mujoco.MjSpec:
    physics = physics or {}
    lighting = lighting or {}
    spec = mujoco.MjSpec()
    spec.modelname = "simeval"
    spec.option.timestep = physics.get("timestep", 0.002)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.iterations = 100
    spec.option.noslip_iterations = physics.get("noslip_iterations", 3)
    spec.memory = 256 * 1024 * 1024
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 4
    spec.visual.headlight.ambient = [0.25] * 3
    spec.visual.headlight.diffuse = [0.35] * 3
    spec.visual.headlight.specular = [0.1] * 3
    spec.default.geom.friction = physics.get("default_friction", [1.6, 0.01, 0.001])
    spec.default.geom.solref = [0.015, 1]
    spec.default.geom.solimp = [0.9, 0.95, 0.001, 0.5, 2]
    sky = spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                           width=512, height=3072)
    sky.rgb1 = [0.78, 0.85, 0.90]
    sky.rgb2 = [0.94, 0.95, 0.93]
    scale = lighting.get("intensity", 1.0)
    shift = np.asarray(lighting.get("dir_jitter", [0, 0, 0]), float)
    spec.worldbody.add_light(pos=[-0.8, 0.2, 2.7], dir=np.r_[0.3, -0.1, -1] + shift,
                             diffuse=np.array([0.55, 0.53, 0.50]) * scale, ambient=np.array([0.25] * 3) * scale)
    spec.worldbody.add_light(pos=[1, -0.6, 2], dir=np.r_[-0.4, 0.2, -1] + shift,
                             diffuse=np.array([0.4, 0.44, 0.48]) * scale, castshadow=False)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.77, 0.80, 0.81, 1])
    return spec


def add_fixed_cameras(spec: mujoco.MjSpec, robot_cfg: dict, jitter: dict | None = None) -> list[CameraSpec]:
    """World-fixed station cameras declared in the robot config (they belong to the station, not to an arm)."""
    cams = []
    jitter = jitter or {}
    for name, c in (robot_cfg.get("fixed_cameras") or {}).items():
        calib = json.loads((ROOT / c["calibration_file"]).read_text())
        j = jitter.get(name, {})
        add_calibrated_camera(
            spec.worldbody, name, calib["cameras"][c.get("calibration_key", name)],
            rotation=c.get("world_rotation"), offset=c.get("world_offset"),
            lower=c.get("lower", 0.0), tilt_up_deg=c.get("tilt_up_deg", 0.0),
            jitter_pos=j.get("pos"), jitter_rot_deg=j.get("rot_deg"),
        )
        cams.append(CameraSpec(name=name, role=name))
    return cams


def compose(robot_cfg: dict, task, variation: dict) -> tuple[mujoco.MjModel, object]:
    """Build the full scene for one episode."""
    spec = base_spec(variation.get("physics"), variation.get("lighting"))
    cameras = add_fixed_cameras(spec, robot_cfg, variation.get("camera_jitter"))
    task.build(spec, variation)  # fixtures first: robots are mounted relative to the world, not to task bodies
    arms, arm_cams = spawn_robot(spec, robot_cfg)
    robot_class = resolve(robot_cfg.get("robot_class", "simeval.sim.robot:MujocoRobot"), {})
    robot = robot_class(arms, cameras + arm_cams)
    model = spec.compile()
    robot.bind(model)
    return model, robot
