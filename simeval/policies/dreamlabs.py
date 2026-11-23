"""Dream Labs (DreamAPI, Dream-1) adapter.

Observation contract follows the Dream Labs YAM MuJoCo example, which is what Dream-1 was validated against in sim:
  * each camera: last 9 frames, 0.2 s apart in simulated time, 320x176 JPEG (q=85), base64;
  * state: per-arm joints (6) and gripper opening (0..1);
  * key style `station` (dream-1-beta; the server computes hand poses from the joints) or `native` (`video.<role>`,
    `state.<part>`; for fine-tunes with an embodiment config).
Actions come back as `q_l_chunk` / `q_r_chunk`, (T, 7) absolute joint targets at 20 Hz.
"""

from __future__ import annotations

import base64
import io
import logging
import time
import warnings

import numpy as np
from PIL import Image

from ..core.types import ActionChunk, Observation
from .base import ApiPolicy

STATION_ROBOT = {
    "robot.arm": "yam",
    "robot.exec_head": "joint",
    "robot.max_joint_vel_rad_s": 0.0,
    "robot.gripper_open": 1.0,
}
YAM_PARTS = {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]}


class DreamLabsPolicy(ApiPolicy):
    def __init__(self, cfg: dict):
        cfg = {"obs_period_s": 0.2, "warmup_s": 1.8, "max_chunk_rows": 24, "api_key_env": "DREAM_API_KEY", **cfg}
        super().__init__(cfg)
        self.checkpoint = cfg.get("checkpoint", "dreamlabs/dream-1-beta")
        self.key_style = cfg.get("key_style", "station")
        self.frame_size = tuple(cfg.get("frame_size", [320, 176]))
        self.jpeg_quality = cfg.get("jpeg_quality", 85)
        self.camera_map = cfg.get("camera_map") or {"top": "top", "left_wrist": "left", "right_wrist": "right"}
        self.arm_map = cfg.get("arm_map") or {"left": "l", "right": "r"}
        self.client = None
        if not self.api_key:
            raise RuntimeError(f"Set {cfg['api_key_env']} (see .env.example).")

    # Session ------------------------------------------------------------------------------------------------------

    def _open(self):
        import dream

        warnings.filterwarnings("ignore", category=dream.DreamSlowNetworkWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="dream")
        logging.getLogger("dream").setLevel(logging.ERROR)
        config = {"checkpoint": self.checkpoint, "timeout_ms": self.cfg.get("timeout_ms", 45000), "strict": False,
                  "action_hz": 20, "_record": False}
        if self.key_style == "native":
            config["embodiment"] = {
                "state": YAM_PARTS, "action": YAM_PARTS,
                "video": {r: f"observation.images.{r}" for r in ("overhead", "wrist_l", "wrist_r")},
            }
        self.client = dream.DreamClient(api_key=self.api_key)
        self.client.policy(config)

    def reset(self, instruction: str, robot_info: dict) -> None:
        super().reset(instruction, robot_info)
        self.close()
        self._open()

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

    # Observation --------------------------------------------------------------------------------------------------

    def _jpeg(self, rgb: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(rgb).resize(self.frame_size, Image.BILINEAR).save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _frame_key(self, cam: str) -> str:
        native = {"top": "overhead", "left": "wrist_l", "right": "wrist_r"}
        return f"image_window.{cam}" if self.key_style == "station" else f"video.{native[cam]}"

    def _state(self, obs: Observation) -> dict:
        out = {"prompt": self.instruction}
        for arm, q in obs.state.items():
            side = self.arm_map[arm]
            if self.key_style == "station":
                out[f"state.arm_{side}.joint_pos"] = q[:6].tolist()
                out[f"state.arm_{side}.gripper_pos"] = q[6:].tolist()
            else:
                name = "left" if side == "l" else "right"
                out[f"state.{name}_arm"] = q[:6].tolist()
                out[f"state.{name}_gripper"] = q[6:].tolist()
        if self.key_style == "station":
            out.update(STATION_ROBOT)
        return out

    def observe(self, obs: Observation) -> None:
        frames = {self._frame_key(self.camera_map[role]): self._jpeg(img)
                  for role, img in obs.images.items() if role in self.camera_map}
        self.client.set_observation(frames, self._state(obs), timestamp=obs.time)

    # Action -------------------------------------------------------------------------------------------------------

    def act(self, obs: Observation) -> ActionChunk:
        reply = None
        for attempt in range(self.max_retries + 1):
            try:
                reply = self.timed(self.client.get_action)
                break
            except Exception as e:  # provider errors: retry what the API says is retryable
                name = type(e).__name__
                wait = getattr(e, "retry_after_s", None)
                if attempt < self.max_retries and (name == "DreamSessionExpired" or wait):
                    time.sleep(float(wait or 0.5))
                    continue
                if self.key_style == "native" and "_eef_" in str(e):
                    self.key_style = "station"  # checkpoint wants hand poses; the station keys carry them
                    self.reset(self.instruction, self.robot_info)
                    self.observe(obs)
                    continue
                raise
        actions = {}
        for arm, side in self.arm_map.items():
            chunk = np.asarray(reply[f"q_{side}_chunk"], float)
            if chunk.ndim != 2 or chunk.shape[1] != 7 or not np.isfinite(chunk).all():
                raise ValueError(f"Dream returned a malformed {arm} chunk: shape {chunk.shape}")
            actions[arm] = chunk
        return ActionChunk(actions, info={"latency_s": self.latencies[-1]})
