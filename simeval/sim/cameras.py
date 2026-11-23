"""Cameras with real intrinsics, plus offscreen rendering.

Policy cameras are calibrated from the physical station (pinhole intrinsics, OpenGL-convention extrinsics), so the
rendered image geometry matches what the model was trained on. That is the largest single lever on the visual sim2real
gap for a pretrained VLA.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class CameraSpec:
    name: str  # MuJoCo camera name in the compiled model
    role: str  # robot-neutral role, e.g. "top", "left_wrist"
    width: int = 640
    height: int = 480
    crop_rows: tuple[int, int] | None = (60, 420)  # station recordings are 640x360 crops of the 640x480 sensor


def _quat_mul(a, b):
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def add_calibrated_camera(body, name: str, entry: dict, rotation=None, offset=None, lower=0.0, tilt_up_deg=0.0,
                          jitter_pos=None, jitter_rot_deg=None):
    """Add a pinhole camera from a calibration entry {position, quaternion_opengl, intrinsics{fx,fy,cx,cy}}."""
    pos = np.asarray(entry["position"], float)
    quat = np.asarray(entry["quaternion_opengl"], float)
    quat = quat / np.linalg.norm(quat)
    if rotation is not None:
        rotation = np.asarray(rotation, float)
        pos = rotation @ pos + np.asarray(offset if offset is not None else [0, 0, 0], float)
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, quat)
        mujoco.mju_mat2Quat(quat, (rotation @ mat.reshape(3, 3)).flatten())
    pos = pos - np.array([0.0, 0.0, lower])
    if tilt_up_deg:
        pitch = np.empty(4)
        mujoco.mju_axisAngle2Quat(pitch, np.array([1.0, 0.0, 0.0]), np.radians(tilt_up_deg))
        quat = _quat_mul(quat, pitch)
    if jitter_pos is not None:
        pos = pos + np.asarray(jitter_pos, float)
    if jitter_rot_deg is not None:
        axis = np.asarray(jitter_rot_deg, float)
        angle = np.linalg.norm(axis)
        if angle > 0:
            dq = np.empty(4)
            mujoco.mju_axisAngle2Quat(dq, axis / angle, np.radians(angle))
            quat = _quat_mul(quat, dq)
    k = entry["intrinsics"]
    cam = body.add_camera(name=name, pos=pos, quat=quat)
    cam.resolution = [640, 480]
    cam.sensor_size = [0.0064, 0.0048]
    cam.focal_pixel = [k["fx"], k["fy"]]
    cam.principal_pixel = [320 - k["cx"], k["cy"] - 240]
    return cam


class Renderer:
    def __init__(self, model: mujoco.MjModel, cameras: list[CameraSpec]):
        self.cameras = cameras
        h = max(c.height for c in cameras)
        w = max(c.width for c in cameras)
        self.renderer = mujoco.Renderer(model, height=h, width=w)

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        out = {}
        for cam in self.cameras:
            self.renderer.update_scene(data, camera=cam.name)
            img = self.renderer.render()
            if cam.crop_rows:
                img = img[cam.crop_rows[0] : cam.crop_rows[1]]
            out[cam.role] = img.copy()
        return out

    def close(self):
        self.renderer.close()
