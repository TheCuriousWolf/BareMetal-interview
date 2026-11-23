"""Aggregation across episodes."""

from __future__ import annotations

import math

import numpy as np


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval: well-behaved at 0/n and n/n, unlike the normal approximation."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(episodes: list[dict], stages: list[str]) -> dict:
    n = len(episodes)
    if n == 0:
        return {"episodes": 0}
    m = [e["metrics"] for e in episodes]
    k = sum(x["success"] for x in m)

    def stat(key):
        vals = [x[key] for x in m if isinstance(x.get(key), (int, float))]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None

    out = {
        "episodes": n,
        "success_rate": k / n,
        "success_ci95": wilson(k, n),
        "progress": stat("progress"),
        "stage_reach_rate": {s: sum(x["stage_times"].get(s) is not None for x in m) / n for s in stages},
        "stage_time_s": {
            s: float(np.mean(t)) if (t := [x["stage_times"][s] for x in m if x["stage_times"].get(s) is not None])
            else None
            for s in stages
        },
    }
    skip = {"success", "failure", "progress", "stage_times"}
    for key in m[0]:
        if key not in skip:
            s = stat(key)
            if s is not None:
                out[key] = s
    return out
