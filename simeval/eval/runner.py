"""Episode loop. Identical for every robot, task and policy.

Timing model:
  * the policy is observed every `obs_period_s` of sim time and queried when its chunk runs out;
  * `latency_mode: hold` pauses physics during inference (isolates policy quality from network speed);
  * `latency_mode: realtime` keeps the robot holding its last target for the measured round trip in sim time, as a
    synchronous controller on the real station would (exposes the cost of slow inference).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .metrics import summarize


def _tile(images: dict[str, np.ndarray], height: int = 180) -> np.ndarray:
    from PIL import Image

    tiles = []
    for role in sorted(images):
        img = Image.fromarray(images[role])
        w = int(img.width * height / img.height)
        tiles.append(np.asarray(img.resize((w, height))))
    return np.concatenate(tiles, axis=1)


def run_episode(env, policy, seed: int, level: str, latency_mode: str = "hold", video_path: Path | None = None,
                log=print) -> dict:
    task = env.task
    variation = env.reset(seed, level)
    policy.reset(task.instruction, {"arms": {m: a.dof for m, a in env.robot.arms.items()},
                                    "cameras": [c.role for c in env.robot.cameras]})
    hz = env.control_hz
    period = max(1, round(policy.spec.obs_period_s * hz))
    max_steps = int(task.max_seconds * hz)
    frames, executed, latencies, clips = [], [], [], []
    step = 0
    wall = time.monotonic()

    def tick(obs=None):
        if step % period == 0:
            obs = obs or env.observe()
            policy.observe(obs)
            if video_path is not None:
                frames.append(_tile(obs.images))

    def advance(targets=None):
        nonlocal step
        if targets is None:
            env.hold()
        else:
            clips.append(max(env.step(targets).values()))
            executed.append(np.concatenate([targets[a] for a in sorted(targets)]))
        step += 1
        tick()

    tick()
    for _ in range(round(policy.spec.warmup_s * hz)):
        advance()
    start_step = step
    outcome = "timeout"
    try:
        while step - start_step < max_steps:
            obs = env.observe()
            t0 = time.monotonic()
            chunk = policy.act(obs)
            latencies.append(chunk.info.get("latency_s", time.monotonic() - t0))
            if latency_mode == "realtime":
                for _ in range(round(latencies[-1] * hz)):
                    advance()
            for i in range(min(len(chunk), policy.spec.max_chunk_rows)):
                advance(chunk.row(i))
                if task.success() or task.failed():
                    break
            if task.success():
                outcome = "success"
                break
            if task.failed():
                outcome = "failed"
                break
    except Exception as e:  # a crashing policy is a failed episode, not a crashed eval
        outcome = f"error: {type(e).__name__}: {e}"
        log(f"    episode error: {outcome}")

    metrics = task.metrics()
    q = np.asarray(executed)
    metrics.update({
        "sim_seconds": float(env.time),
        "wall_seconds": time.monotonic() - wall,
        "queries": len(latencies),
        "latency_mean_s": float(np.mean(latencies)) if latencies else None,
        "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else None,
        # Share of executed steps whose target left the joint limits by > 0.01 rad, and the worst overshoot.
        "joint_limit_clip_rate": float(np.mean(np.asarray(clips) > 0.01)) if clips else 0.0,
        "joint_limit_clip_max_rad": float(np.max(clips)) if clips else 0.0,
        "action_jerk": float(np.mean(np.abs(np.diff(q, 2, axis=0)))) if len(q) > 2 else 0.0,
    })
    if video_path is not None and frames:
        import imageio.v2 as imageio

        imageio.mimsave(video_path, frames, fps=int(round(1 / policy.spec.obs_period_s)) * 2, macro_block_size=1)
    return {"seed": seed, "level": level, "outcome": outcome, "metrics": metrics, "variation": variation}


def run_eval(env, policy, episodes: int, seed: int, level: str, out_dir: Path, latency_mode: str = "hold",
             video: bool = True, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for ep in range(episodes):
        s = seed + ep
        log(f"  episode {ep + 1}/{episodes} seed={s} level={level}")
        r = run_episode(env, policy, s, level, latency_mode, out_dir / f"episode_{ep:03d}.mp4" if video else None, log)
        m = r["metrics"]
        log(f"    {r['outcome']}  progress={m['progress']:.2f} "
            f"stages={[k for k, v in m['stage_times'].items() if v is not None]} queries={m['queries']}")
        (out_dir / f"episode_{ep:03d}.json").write_text(json.dumps(r, indent=2, default=float))
        results.append(r)
    policy.close()
    env.close()
    summary = {"policy": policy.describe(), "task": env.task.name, "level": level, "latency_mode": latency_mode,
               **summarize(results, env.task.stages)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
