"""Record an annotated video of one evaluation episode.

    python scripts/record_demo.py configs/evals/sponge_dream.yaml --seed 1004 --level pose --out runs/demo

Writes <video>.metadata.json describing the run (robot, mounts, gripper, cameras, every interactable and its sampled
parameters, the instruction, policy, metrics, stages and end conditions) and the video itself at real time
(20 fps = control rate):
a wide overview camera, the three policy cameras exactly as the model sees them, and a side panel with the stages
reached and the live metric values. It uses the same episode loop as run_eval.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from scripts.run_eval import load_env_file  # noqa: E402
from simeval.eval.runner import run_episode  # noqa: E402
from simeval.registry import make_policy, make_task  # noqa: E402
from simeval.sim.sim_env import SimEnv  # noqa: E402

W, H = 1280, 720
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MONO_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def font(size, mono=False):
    try:
        return ImageFont.truetype(MONO_PATH if mono else FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def fmt(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return str(v)


class DemoRecorder:
    def __init__(self, robot_cfg, task_cfg, policy_cfg, seed, level):
        self.robot_cfg, self.task_cfg, self.policy_cfg = robot_cfg, task_cfg, policy_cfg
        self.seed, self.level = seed, level
        self.frames = []
        self.renderer = None
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = (0.0, 0.02, 0.80)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.25, -90.0, -38.0
        self.f_title, self.f_body, self.f_small, self.f_mono = font(30), font(17), font(14), font(14, True)

    # Per step -------------------------------------------------------------------------------------------------------
    def __call__(self, env, event):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(env.model, height=540, width=960)
        self.renderer.update_scene(env.data, camera=self.cam)
        overview = self.renderer.render()
        canvas = Image.new("RGB", (W, H), (15, 17, 22))
        canvas.paste(Image.fromarray(overview), (0, 0))
        cams = env.renderer.render(env.data)
        for i, role in enumerate(["top", "left_wrist", "right_wrist"]):
            if role in cams:
                tile = Image.fromarray(cams[role]).resize((320, 180))
                canvas.paste(tile, (i * 320, 540))
        d = ImageDraw.Draw(canvas)
        for i, role in enumerate(["top", "left_wrist", "right_wrist"]):
            d.text((i * 320 + 6, 544), f"policy camera: {role}", font=self.f_small, fill="white")
        task = env.task
        d.rectangle([(0, 0), (960, 34)], fill=(15, 17, 22))
        state = "filling the model's 9-frame history" if event == "warmup" else "model controlling the arms"
        d.text((10, 7), f"t = {env.time:5.1f} s   |   {state}   |   \"{task.instruction}\"", font=self.f_body, fill="white")

        # Side panel: stages and metrics.
        x, y = 975, 12
        d.text((x, y), "Stages", font=self.f_title, fill="white")
        y += 44
        for st, t in task.stage_times.items():
            done = t is not None
            d.text((x, y), ("✔ " if done else "○ ") + st, font=self.f_body, fill=(120, 230, 120) if done else (150, 150, 150))
            if done:
                d.text((x + 200, y + 2), f"{t:.1f} s", font=self.f_small, fill=(120, 230, 120))
            y += 26
        y += 14
        d.text((x, y), "Metrics (live)", font=self.f_title, fill="white")
        y += 44
        for name, m in task.metrics_now.items():
            d.text((x, y), f"{name[:20]:20s} {m.value:7.3f}", font=self.f_mono, fill=(220, 220, 220))
            y += 20
        hit = sum(t is not None for t in task.stage_times.values())
        d.text((x, H - 80), f"Stages hit {hit}/{len(task.stage_times)}", font=self.f_title, fill="white")
        if task.success():
            d.text((x, H - 40), "SUCCESS", font=self.f_title, fill=(120, 230, 120))
        elif task.failed():
            d.text((x, H - 40), "FAILURE", font=self.f_title, fill=(240, 110, 110))
        self.frames.append(np.asarray(canvas))


def describe_run(env, robot_cfg, task_cfg, policy_cfg, args) -> dict:
    """Everything needed to understand the run: what was built, what was asked, how it is scored."""
    task, robot = env.task, env.robot
    objects = {}
    for name, obj in task.objects.items():
        entry = {"type": obj.type, "sampled": env.variation["objects"][name]}
        if obj.type == "spill":
            entry.update(decals=len(obj.cells), erased_by=obj.params["erased_by"],
                         min_press_force_N=obj.params.get("min_press_force"), grasp_required=obj.params.get("require_grasp"))
        if hasattr(obj, "size"):
            entry["size_cm"] = [round(2 * x * 100, 2) for x in obj.size]
        objects[name] = entry
    return {
        "task": task.name, "instruction": task.instruction, "time_limit_s": task.max_seconds,
        "seed": args.seed, "randomization": args.level, "control_hz": env.control_hz,
        "robot": {"model": robot_cfg["robot_model"], "class": robot_cfg.get("robot_class", "MujocoRobot"),
                  "gripper": robot_cfg.get("gripper"), "mounts": robot_cfg["mounts"],
                  "tcp_sites": [a.tcp_site for a in robot.arms.values()], "cameras": [c.role for c in robot.cameras]},
        "policy": {k: v for k, v in policy_cfg.items() if "key" not in k.lower()},
        "objects": objects,
        "markers": [env.model.site(i).name for i in range(env.model.nsite) if "." in env.model.site(i).name],
        "metrics": task_cfg.get("metrics", {}), "stages": task_cfg.get("stages", []), "end": task_cfg.get("end", {}),
        "video_layout": "top-left: overview camera; bottom: policy cameras top, left_wrist, right_wrist (as the model "
                        "sees them); right: stages reached with times and live metric values",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--seed", type=int, default=1004)
    ap.add_argument("--level", default="pose")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--policy", default=None, help="policy config path to use instead of the eval's")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "demo")
    args = ap.parse_args()
    load_env_file(ROOT / ".env")
    cfg = yaml.safe_load(args.config.read_text())
    robot_cfg = yaml.safe_load((ROOT / cfg["robot"]).read_text())
    task_cfg = yaml.safe_load((ROOT / cfg["task"]).read_text())
    if args.seconds:
        task_cfg["max_seconds"] = args.seconds
    policy_cfg = yaml.safe_load((ROOT / (args.policy or cfg["policy"])).read_text())
    task, policy = make_task(task_cfg), make_policy(policy_cfg)
    env = SimEnv(robot_cfg, task, control_hz=cfg.get("control_hz", 20))
    rec = DemoRecorder(robot_cfg, task_cfg, policy_cfg, args.seed, args.level)

    # Intro: build the episode's scene once to describe it (same seed -> same scene as the run).
    env.reset(args.seed, args.level)
    metadata = describe_run(env, robot_cfg, task_cfg, policy_cfg, args)

    seen = {"model": None}

    def on_step(env_, event):
        if seen["model"] is not env_.model:  # run_episode rebuilt the scene: renderer must follow the new model
            if rec.renderer is not None:
                rec.renderer.close()
            rec.renderer, seen["model"] = None, env_.model
        rec(env_, event)

    result = run_episode(env, policy, args.seed, args.level, cfg.get("latency_mode", "hold"), None, on_step=on_step)
    policy.close()
    env.close()
    args.out.mkdir(parents=True, exist_ok=True)
    rec.frames += [rec.frames[-1]] * 40  # hold the last frame 2 s
    import imageio.v2 as imageio

    video = args.out / f"{task.name}_{policy.spec.name}_seed{args.seed}_{args.level}.mp4"
    imageio.mimsave(video, rec.frames, fps=20, macro_block_size=1, quality=7)
    metadata["result"] = result
    (args.out / (video.stem + ".metadata.json")).write_text(json.dumps(metadata, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    sc = result["score"]
    print(f"{sc['result']} ({sc['why']})  stages hit {sc['stages_hit']}: {', '.join(sc['hit']) or '-'}")
    print(f"video:    {video}\nmetadata: {args.out / (video.stem + '.metadata.json')}")


if __name__ == "__main__":
    main()
