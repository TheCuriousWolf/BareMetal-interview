"""Seeded parameter sampling shared by every task and scene file.

A scene parameter is either a plain value (fixed) or a record

    {nominal: <value>, range: [lo, hi]}      # uniform in [lo, hi]
    {nominal: <value>, jitter: <half-width>} # nominal ± half-width (scalar or per-element)
    ... plus  group: <name>

The eval config picks a randomization level; each level lists the groups that are sampled. Parameters in other
groups take their nominal value. Same seed + same level → same scene, so policies are compared on identical episodes.
"""

from __future__ import annotations

import numpy as np

DEFAULT_LEVELS = {
    "none": [],
    "pose": ["pose"],
    "full": ["pose", "shape", "physics", "lighting", "camera", "appearance"],
}


class Sampler:
    def __init__(self, rng: np.random.Generator, groups: list[str]):
        self.rng = rng
        self.groups = set(groups)
        self.record: dict[str, object] = {}

    def __call__(self, key: str, value):
        out = self._sample(value)
        self.record[key] = out.tolist() if isinstance(out, np.ndarray) else out
        return out

    def _sample(self, value):
        if not isinstance(value, dict) or "nominal" not in value:
            return np.asarray(value, float) if isinstance(value, list) else value
        nominal = np.asarray(value["nominal"], float)
        if value.get("group", "pose") not in self.groups:
            return nominal if nominal.ndim else float(nominal)
        if "range" in value:
            lo, hi = (np.asarray(v, float) for v in value["range"])
            out = self.rng.uniform(lo, hi)
        elif "jitter" in value:
            j = np.asarray(value["jitter"], float)
            out = nominal + self.rng.uniform(-j, j, size=nominal.shape)
        elif "choices" in value:
            out = np.asarray(value["choices"][self.rng.integers(len(value["choices"]))], float)
        else:
            out = nominal
        return out if np.ndim(out) else float(out)
