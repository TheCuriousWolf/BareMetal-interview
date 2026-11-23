"""Run an evaluation from a config.

    python scripts/run_eval.py configs/evals/sponge_dream.yaml
    python scripts/run_eval.py configs/evals/sponge_dream.yaml --set episodes=3 level=full policy=configs/policies/hold.yaml
    python scripts/run_eval.py configs/evals/sponge_dream.yaml --set policy.checkpoint=dream-1-ft:my-run task.max_seconds=60

An eval config names a robot, a task and a policy config (paths) plus run settings. `--set a.b=v` overrides any field;
`robot.*`, `task.*` and `policy.*` reach into the referenced configs. Values are parsed as YAML.
"""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from simeval.eval.runner import run_eval  # noqa: E402
from simeval.registry import make_policy, make_task  # noqa: E402
from simeval.sim.sim_env import SimEnv  # noqa: E402


def load_env_file(path: Path) -> None:
    if path.is_file():
        for line in path.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip())


def set_path(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    load_env_file(ROOT / ".env")

    cfg = yaml.safe_load(args.config.read_text())
    top_overrides = [o for o in args.set if o.split("=", 1)[0] in ("robot", "task", "policy")]
    for o in top_overrides:  # swap a whole component first
        k, v = o.split("=", 1)
        cfg[k] = v
    parts = {k: yaml.safe_load((ROOT / cfg[k]).read_text()) for k in ("robot", "task", "policy")}
    for o in args.set:
        if o in top_overrides:
            continue
        k, v = o.split("=", 1)
        head = k.split(".", 1)[0]
        if head in parts and "." in k:
            set_path(parts[head], k.split(".", 1)[1], yaml.safe_load(v))
        else:
            set_path(cfg, k, yaml.safe_load(v))

    task = make_task(parts["task"])
    policy = make_policy(parts["policy"])
    env = SimEnv(parts["robot"], task, control_hz=cfg.get("control_hz", 20))
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or ROOT / "runs" / f"{cfg.get('name', 'eval')}_{policy.spec.name}_{cfg['level']}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"eval": cfg, **parts, "policy": policy.describe()}, indent=2))
    print(f"{task.name} | robot={parts['robot']['robot_model']} mounts={list(parts['robot']['mounts'])} | "
          f"policy={policy.spec.name} | level={cfg['level']} | {cfg['episodes']} episodes -> {out}")
    summary = run_eval(env, policy, cfg["episodes"], cfg["seed"], cfg["level"], out,
                       cfg.get("latency_mode", "hold"), cfg.get("video", True))
    print(json.dumps({k: summary[k] for k in ("success_rate", "success_ci95", "progress", "stage_reach_rate")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
