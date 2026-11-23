"""Policy interface: the only thing a model provider has to implement.

The runner talks to every policy the same way:

    policy.reset(instruction, robot_info)    # new episode
    policy.observe(obs)                     # every `spec.obs_period_s` of sim time (frame history, streaming APIs)
    chunk = policy.act(obs)                 # when the previous chunk is used up
    policy.close()

`ApiPolicy` and `LocalPolicy` add the plumbing that remote and on-device models need (key, endpoint, retries, latency
timing; checkpoint, device, load). A provider adapter only translates our Observation/ActionChunk to its own format.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

from ..core.types import ActionChunk, Observation, PolicySpec


class Policy(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.spec = PolicySpec(
            name=cfg.get("name", type(self).__name__),
            obs_period_s=cfg.get("obs_period_s", 0.2),
            warmup_s=cfg.get("warmup_s", 0.0),
            max_chunk_rows=cfg.get("max_chunk_rows", 24),
        )
        # Robot-role -> model-key mappings; every adapter reads them from config instead of hard-coding names.
        self.camera_map: dict[str, str] = cfg.get("camera_map", {})
        self.arm_map: dict[str, str] = cfg.get("arm_map", {})

    def reset(self, instruction: str, robot_info: dict) -> None:
        self.instruction = instruction
        self.robot_info = robot_info

    def observe(self, obs: Observation) -> None:
        """Called at the observation rate. Default: nothing (single-frame policies only need act())."""

    @abstractmethod
    def act(self, obs: Observation) -> ActionChunk: ...

    def close(self) -> None:
        pass

    def describe(self) -> dict:
        return {"name": self.spec.name, "class": type(self).__name__,
                **{k: v for k, v in self.cfg.items() if "key" not in k.lower()}}


class ApiPolicy(Policy):
    """Base for hosted models. Reads the key from an environment variable named in config, never from the config."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.endpoint = cfg.get("endpoint")
        self.checkpoint = cfg.get("checkpoint")
        self.api_key = os.environ.get(cfg.get("api_key_env", "API_KEY"), "")
        self.max_retries = cfg.get("max_retries", 3)
        self.latencies: list[float] = []

    def timed(self, fn, *args, **kwargs):
        t = time.monotonic()
        out = fn(*args, **kwargs)
        self.latencies.append(time.monotonic() - t)
        return out


class LocalPolicy(Policy):
    """Base for models run in-process (torch, jax, onnx). Subclasses implement load() and predict()."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.checkpoint = cfg.get("checkpoint")
        self.device = cfg.get("device", "cuda")
        self.model = None

    def reset(self, instruction: str, robot_info: dict) -> None:
        super().reset(instruction, robot_info)
        if self.model is None:
            self.model = self.load()

    @abstractmethod
    def load(self): ...
