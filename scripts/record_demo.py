"""Record an annotated video of one evaluation episode.

    python scripts/record_demo.py configs/evals/sponge_dream.yaml --seed 1004 --level pose --out runs/demo

The video opens with a card describing the scene (robot, mounts, gripper, cameras, every interactable and its sampled
parameters, the instruction and the policy), then plays the episode at real time (20 fps = control rate):
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

    # Intro card -----------------------------------------------------------------------------------------------------
    def intro(self, env, variation):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(env.model, height=540, width=960)
        self.renderer.update_scene(env.data, camera=self.cam)
        bg = Image.fromarray(self.renderer.render()).resize((W, H))
        bg = Image.blend(bg, Image.new("RGB", (W, H), (15, 17, 22)), 0.72)
        d = ImageDraw.Draw(bg)
        task, robot = env.task, env.robot
        d.text((40, 28), f"Task: {task.name}", font=self.f_title, fill="white")
        d.text((40, 70), f'Instruction to the model: "{task.instruction}"', font=self.f_body, fill=(255, 220, 120))
        y = 112
        mounts = ", ".join(f"{m} at {fmt(self.robot_cfg['mounts'][m]['pos'])}" for m in robot.arms)
        lines = [
            ("Robot", f"{self.robot_cfg['robot_model']} (URDF), {len(robot.arms)} arm(s): {mounts}"),
            ("Gripper", f"{self.robot_cfg.get('gripper')} (parallel, opening 0..1), TCP sites: "
                        + ", ".join(a.tcp_site for a in robot.arms.values())),
            ("Cameras", ", ".join(c.role for c in robot.cameras) + "  (calibrated on the real station)"),
            ("Policy", f"{self.policy_cfg.get('name')}  checkpoint={self.policy_cfg.get('checkpoint', '-')}  "
                       f"{self.policy_cfg.get('max_chunk_rows')} actions per query"),
            ("Episode", f"seed {self.seed}, randomization '{self.level}', limit {task.max_seconds:.0f} s, 20 Hz control"),
        ]
        for k, v in lines:
            d.text((40, y), k, font=self.f_body, fill=(140, 200, 255))
            d.text((150, y), v, font=self.f_body, fill="white")
            y += 28
        y += 10
        d.text((40, y), "Interactables spawned (sampled values for this seed):", font=self.f_body, fill=(140, 200, 255))
        y += 30
        for name, obj in task.objects.items():
            vals = variation["objects"][name]
            shown = {k: v for k, v in vals.items() if k not in ("phases", "rgba")}
            extra = ""
            if obj.type == "spill":
                extra = f"  decals={len(obj.cells)}  erased_by={obj.params['erased_by']}  " \
                        f"min_press={obj.params.get('min_press_force')} N  grasp required={obj.params.get('require_grasp')}"
            if obj.type in ("sponge", "object"):
                extra = f"  size={fmt([round(2 * x * 100, 1) for x in obj.size])} cm"
            d.text((60, y), f"{name} [{obj.type}]", font=self.f_body, fill=(255, 220, 120))
            d.text((260, y), "  ".join(f"{k}={fmt(v)}" for k, v in shown.items()) + extra, font=self.f_small, fill="white")
            y += 26
        y += 10
        d.text((40, y), "Success: " + " and ".join(task.success_when) + "     Failure: " + ", ".join(task.failure_when),
               font=self.f_body, fill=(160, 255, 160))
        y += 28
        d.text((40, y), "Stages: " + "  ->  ".join(task.stages), font=self.f_body, fill="white")
        card = np.asarray(bg)
        self.frames += [card] * 100  # 5 s

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
        if task.success():
            d.text((x, H - 40), "SUCCESS", font=self.f_title, fill=(120, 230, 120))
        elif task.failed():
            d.text((x, H - 40), "FAILED", font=self.f_title, fill=(240, 110, 110))
        self.frames.append(np.asarray(canvas))


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
    rec.intro(env, env.variation)
    rec.renderer.close()
    rec.renderer = None

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
    rec.frames += [rec.frames[-1]] * 60  # hold the last frame 3 s
    import imageio.v2 as imageio

    video = args.out / f"{task.name}_{policy.spec.name}_seed{args.seed}_{args.level}.mp4"
    imageio.mimsave(video, rec.frames, fps=20, macro_block_size=1, quality=7)
    (args.out / (video.stem + ".json")).write_text(json.dumps(result, indent=2, default=float))
    m = result["metrics"]
    print(f"{result['outcome']}  progress={m['progress']:.2f}  stages={m['stage_times']}")
    print(f"video: {video}")


if __name__ == "__main__":
    main()
