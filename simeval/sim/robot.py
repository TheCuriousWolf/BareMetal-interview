"""Spawning robots from robot_models/ and driving them generically.

A robot config (configs/robots/*.yaml) names a robot model folder, a gripper folder and one or more mounts:

    robot_model: yam
    gripper: yam_linear
    mounts:
      left:  {pos: [0.305, 0.24, 0.772], quat: [...]}
      right: {pos: [-0.305, 0.24, 0.772], quat: [...]}

Each mount gets its own copy of the arm (URDF or MJCF) with the gripper attached on the flange, and every name in that
copy is prefixed with the mount name (`left_joint1`, `left_tcp`, ...). One mount gives a single-arm robot.
After compilation, `Robot` reads state and writes targets using only names, so nothing here is YAM-specific.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

from ..paths import ROBOT_MODELS
from .cameras import CameraSpec, add_calibrated_camera


@dataclass
class ArmHandle:
    """Names (already prefixed) of everything one mounted arm exposes."""

    mount: str
    joints: list[str]
    actuators: list[str]
    gripper_joints: list[str]
    gripper_actuators: list[str]
    open_ctrl: float
    closed_ctrl: float
    open_qpos_sum: float
    tcp_site: str
    finger_bodies: list[str]
    home: np.ndarray  # joints then gripper opening

    @property
    def dof(self) -> int:
        return len(self.joints) + (1 if self.gripper_actuators else 0)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _arm_spec(model_dir: Path, cfg: dict) -> mujoco.MjSpec:
    """Load the arm description (URDF or MJCF) as a fresh MjSpec."""
    if "urdf" in cfg:
        path = model_dir / cfg["urdf"]
        text = path.read_text()
        # Resolve mesh paths relative to the URDF and keep visual geometry.
        text = re.sub(
            r'filename="([^"]+)"',
            lambda m: f'filename="{(path.parent / m.group(1).replace("package://", "")).resolve()}"',
            text,
        )
        # Exported URDFs often reference meshes that were never shipped; drop those geometry blocks.
        text = re.sub(
            r"<(visual|collision)>.*?</\1>",
            lambda m: m.group(0) if all(Path(f).exists() for f in re.findall(r'filename="([^"]+)"', m.group(0)))
            else "",
            text,
            flags=re.S,
        )
        text = re.sub(
            r"(<robot[^>]*>)",
            r'\1<mujoco><compiler discardvisual="false" fusestatic="false" strippath="false"/></mujoco>',
            text,
            count=1,
        )
        return mujoco.MjSpec.from_string(text)
    return mujoco.MjSpec.from_file(str(model_dir / cfg["mjcf"]))


def spawn_robot(spec: mujoco.MjSpec, robot_cfg: dict) -> tuple[list[ArmHandle], list[CameraSpec]]:
    """Add every mounted arm (with gripper and robot cameras) to the scene spec."""
    model_dir = ROBOT_MODELS / robot_cfg["robot_model"]
    model_cfg = load_yaml(model_dir / "robot.yaml")
    gripper_name = robot_cfg.get("gripper", model_cfg.get("default_gripper"))
    gripper_dir = ROBOT_MODELS / "grippers" / gripper_name if gripper_name else None
    gripper_cfg = load_yaml(gripper_dir / "gripper.yaml") if gripper_dir else None

    arms, cameras = [], []
    for mount, pose in robot_cfg["mounts"].items():
        arm = _arm_spec(model_dir, model_cfg)
        p = f"{mount}_"
        act = model_cfg.get("actuator", {})
        for jname in model_cfg["joints"]:
            joint = arm.joint(jname)
            joint.damping = model_cfg.get("joint_damping", joint.damping)
            joint.armature = model_cfg.get("joint_armature", joint.armature)
            a = arm.add_actuator(name=jname, target=jname, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=act.get("kp", 100.0), kv=act.get("kv", 10.0))
            a.ctrllimited = True
            a.ctrlrange = joint.range
            if "force_limit" in act:
                a.forcelimited = True
                a.forcerange = [-act["force_limit"], act["force_limit"]]
        flange = arm.body(model_cfg["flange_link"])
        if gripper_cfg is not None:
            gspec = mujoco.MjSpec.from_file(str(gripper_dir / gripper_cfg["mjcf"]))
            gm = model_cfg.get("gripper_mount", {})
            frame = flange.add_frame(pos=gm.get("pos", [0, 0, 0]), quat=gm.get("quat", [1, 0, 0, 0]))
            frame.attach_body(gspec.body(gripper_cfg["root_body"]), "", "")
        if "rgba" in model_cfg:  # URDF colours are often placeholders; use the real finish
            for geom in arm.geoms:
                geom.rgba = model_cfg["rgba"]
                geom.material = ""
        for a, b in model_cfg.get("contact_excludes", []):
            arm.add_exclude(bodyname1=a, bodyname2=b)
        if model_cfg.get("gravity_compensation"):
            for body in arm.bodies:
                if body.name != "world":
                    body.gravcomp = 1.0
        for cam_name, cam in (model_cfg.get("cameras") or {}).items():
            calib = json.loads((model_dir / cam["calibration_file"]).read_text())
            entry = calib["cameras"][cam["calibration_key"].format(mount=mount)]
            add_calibrated_camera(arm.body(cam["parent"]), cam_name, entry)
            cameras.append(CameraSpec(name=f"{p}{cam_name}", role=f"{mount}_{cam_name}"))
        frame = spec.worldbody.add_frame(pos=pose["pos"], quat=pose.get("quat", [1, 0, 0, 0]))
        frame.attach_body(arm.body(model_cfg["base_link"]), p, "")
        # Excludes declared inside the gripper file do not survive attach; re-add finger-finger excludes.
        if gripper_cfg is not None and len(gripper_cfg["finger_bodies"]) == 2:
            a, b = gripper_cfg["finger_bodies"]
            spec.add_exclude(bodyname1=p + a, bodyname2=p + b)
        # A gripper from another vendor can overlap the convex hulls of the wrist links it sits next to.
        if gripper_cfg is not None and gripper_name != model_cfg.get("default_gripper"):
            for link in model_cfg.get("gripper_exclude_links", []):
                for gb in [gripper_cfg["root_body"], *gripper_cfg["finger_bodies"]]:
                    spec.add_exclude(bodyname1=p + link, bodyname2=p + gb)
        home = list(robot_cfg.get("home", {}).get(mount, model_cfg.get("home", [0.0] * len(model_cfg["joints"]))))
        arms.append(
            ArmHandle(
                mount=mount,
                joints=[p + j for j in model_cfg["joints"]],
                actuators=[p + j for j in model_cfg["joints"]],
                gripper_joints=[p + j for j in gripper_cfg["joints"]] if gripper_cfg else [],
                gripper_actuators=[p + j for j in gripper_cfg["actuators"]] if gripper_cfg else [],
                open_ctrl=gripper_cfg["open_ctrl"] if gripper_cfg else 0.0,
                closed_ctrl=gripper_cfg.get("closed_ctrl", 0.0) if gripper_cfg else 0.0,
                open_qpos_sum=gripper_cfg["open_qpos_sum"] if gripper_cfg else 1.0,
                tcp_site=p + (gripper_cfg["tcp_site"] if gripper_cfg else model_cfg["tcp_site"]),
                finger_bodies=[p + b for b in gripper_cfg["finger_bodies"]] if gripper_cfg else [],
                home=np.asarray(home + ([1.0] if gripper_cfg else []), float),
            )
        )
    return arms, cameras


class Robot(ABC):
    """What the simulator, tasks and policies need from a robot. Override these for robots that config cannot
    describe (coupled joints, mobile bases, dexterous hands); the default MujocoRobot covers URDF/MJCF arms."""

    arms: dict[str, ArmHandle]  # by mount name
    cameras: list[CameraSpec]
    body_ids: set[int]  # every robot body
    finger_ids: dict[str, set[int]]  # gripper finger bodies per arm

    @abstractmethod
    def bind(self, model: mujoco.MjModel) -> None: ...

    @abstractmethod
    def reset(self, data: mujoco.MjData) -> None: ...

    @abstractmethod
    def state(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """Per arm: joint positions then gripper opening (0 closed .. 1 open)."""

    @abstractmethod
    def apply(self, data: mujoco.MjData, targets: dict[str, np.ndarray]) -> dict[str, float]:
        """Send absolute joint + gripper targets. Returns how far each arm's targets were clipped."""

    @abstractmethod
    def fk(self, data: mujoco.MjData, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """End-effector (TCP) position and quaternion in the world frame."""

    @abstractmethod
    def ik(self, data: mujoco.MjData, arm: str, pos, quat=None) -> np.ndarray:
        """Joint positions that put the TCP at pos (and quat, if given)."""

    def sensors(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """Extra readings a policy may use. Default: joint velocities and actuator efforts per arm."""
        return {}

    last_action: dict[str, np.ndarray] | None = None


class MujocoRobot(Robot):
    """Default robot: any arm described by robot_models/<name>/robot.yaml and spawned by spawn_robot()."""

    def __init__(self, arms: list[ArmHandle], cameras: list[CameraSpec]):
        self.arms = {a.mount: a for a in arms}
        self.cameras = cameras
        self.last_action = None
    def bind(self, model: mujoco.MjModel) -> None:
        self.model = model
        prefixes = tuple(f"{m}_" for m in self.arms)
        self.body_ids = {i for i in range(model.nbody) if model.body(i).name.startswith(prefixes)}
        self.finger_ids = {m: {model.body(b).id for b in a.finger_bodies} for m, a in self.arms.items()}

    def limits(self, mount: str) -> np.ndarray:
        arm = self.arms[mount]
        rows = [self.model.joint(j).range for j in arm.joints]
        if arm.gripper_actuators:
            rows.append([0.0, 1.0])
        return np.asarray(rows, float)

    def reset(self, data: mujoco.MjData) -> None:
        for mount, arm in self.arms.items():
            for j, v in zip(arm.joints, arm.home):
                data.joint(j).qpos[0] = v
            if arm.gripper_joints:
                per = arm.home[-1] * arm.open_qpos_sum / len(arm.gripper_joints)
                for j in arm.gripper_joints:
                    data.joint(j).qpos[0] = per
            self.apply(data, {mount: arm.home})

    def state(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        out = {}
        for mount, arm in self.arms.items():
            q = [data.joint(j).qpos[0] for j in arm.joints]
            if arm.gripper_joints:
                opening = sum(data.joint(j).qpos[0] for j in arm.gripper_joints) / arm.open_qpos_sum
                q.append(float(np.clip(opening, 0.0, 1.0)))
            out[mount] = np.asarray(q, float)
        return out

    def apply(self, data: mujoco.MjData, targets: dict[str, np.ndarray]) -> dict[str, float]:
        """Write absolute targets (joints, gripper 0..1), clipped to limits. Returns the largest clip per arm."""
        clipped = {}
        for mount, target in targets.items():
            arm = self.arms[mount]
            target = np.asarray(target, float)
            if target.shape != (arm.dof,) or not np.isfinite(target).all():
                raise ValueError(f"{mount}: expected {arm.dof} finite values, got shape {target.shape}")
            lim = self.limits(mount)
            bounded = np.clip(target, lim[:, 0], lim[:, 1])
            self.last_action = {**(self.last_action or {}), mount: bounded}
            clipped[mount] = float(np.abs(bounded - target).max())
            for a, v in zip(arm.actuators, bounded):
                data.actuator(a).ctrl[0] = v
            if arm.gripper_actuators:
                ctrl = arm.closed_ctrl + bounded[-1] * (arm.open_ctrl - arm.closed_ctrl)
                for a in arm.gripper_actuators:
                    data.actuator(a).ctrl[0] = ctrl
        return clipped

    def fk(self, data, arm):
        site = data.site(self.arms[arm].tcp_site)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, site.xmat)
        return site.xpos.copy(), quat

    def ik(self, data, arm, pos, quat=None, iters: int = 100, damping: float = 1e-3) -> np.ndarray:
        """Damped least squares on a scratch copy of the state, from the current joints; respects joint limits."""
        m, a = self.model, self.arms[arm]
        scratch = mujoco.MjData(m)
        scratch.qpos[:] = data.qpos
        dofs = [m.joint(j).dofadr[0] for j in a.joints]
        qadr = [m.joint(j).qposadr[0] for j in a.joints]
        lim = self.limits(arm)[: len(a.joints)]
        site = m.site(a.tcp_site).id
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, scratch)
            mujoco.mj_comPos(m, scratch)
            err = np.asarray(pos, float) - scratch.site_xpos[site]
            mujoco.mj_jacSite(m, scratch, jp, jr, site)
            J = jp[:, dofs]
            if quat is not None:
                cur = np.empty(4)
                mujoco.mju_mat2Quat(cur, scratch.site_xmat[site])
                dq = np.empty(3)
                mujoco.mju_subQuat(dq, np.asarray(quat, float), cur)
                err, J = np.r_[err, dq], np.r_[J, jr[:, dofs]]
            if np.linalg.norm(err) < 1e-4:
                break
            step = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(len(err)), err)
            scratch.qpos[qadr] = np.clip(scratch.qpos[qadr] + step, lim[:, 0], lim[:, 1])
        return scratch.qpos[qadr].copy()

    def sensors(self, data):
        out = {}
        for mount, a in self.arms.items():
            out[f"{mount}.joint_vel"] = np.array([data.joint(j).qvel[0] for j in a.joints])
            out[f"{mount}.effort"] = np.array([data.actuator(x).force[0] for x in a.actuators])
            out[f"{mount}.eef_pos"], out[f"{mount}.eef_quat"] = self.fk(data, mount)
        return out
