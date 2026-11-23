"""Checks that the scoring measures what it claims and that robots, grippers and tasks swap by config alone."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from simeval.registry import make_policy, make_task  # noqa: E402
from simeval.sim.sim_env import SimEnv  # noqa: E402


def cfg(path):
    return yaml.safe_load((ROOT / path).read_text())


def sponge_env(require_grasp=True, robot="configs/robots/yam_bimanual.yaml"):
    t = cfg("configs/tasks/sponge_scrub.yaml")
    t["scene_overrides"] = {"scoring": {"require_grasp": require_grasp}}
    env = SimEnv(cfg(robot), make_task(t))
    env.reset(seed=0, level="none")
    return env


def scrub(env, press_newtons, strokes=6):
    """Drive the sponge back and forth across the stain, optionally pushing it down."""
    m, d, task = env.model, env.data, env.task
    jnt = m.joint("sponge_free")
    adr, dof = jnt.qposadr[0], jnt.dofadr[0]
    cx, cy = task.cell_centers.mean(axis=0)
    z = task.counter_top + task.sponge_half[2]
    for k in range(strokes * 20):
        x = cx + 0.05 * np.sin(k / 20 * np.pi)
        d.qpos[adr : adr + 3] = [x, cy, z]
        d.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        d.qvel[dof : dof + 6] = 0
        d.xfrc_applied[task.sponge_id] = [0, 0, -press_newtons, 0, 0, 0]
        env.hold()
    return task.metrics()


def test_scrubbing_needs_pressure_and_motion():
    pressed = scrub(sponge_env(require_grasp=False), press_newtons=5.0)
    light = scrub(sponge_env(require_grasp=False), press_newtons=0.0)
    assert pressed["dirt_removed"] > 0.2 and pressed["stage_times"]["stain_contact"] is not None
    assert light["dirt_removed"] == 0.0  # resting weight alone is below min_press_force


def test_loose_sponge_does_not_count_when_grasp_required():
    assert scrub(sponge_env(require_grasp=True), press_newtons=5.0)["dirt_removed"] == 0.0


def test_hold_policy_scores_zero():
    env = sponge_env()
    policy = make_policy(cfg("configs/policies/hold.yaml"))
    policy.reset(env.task.instruction, {})
    for _ in range(40):
        chunk = policy.act(env.observe(render=False))
        env.step(chunk.row(0))
    m = env.task.metrics()
    assert m["progress"] == 0.0 and not m["success"]


def test_single_arm_mount():
    env = sponge_env(robot="configs/robots/yam_single.yaml")
    assert list(env.robot.arms) == ["right"]
    assert set(env.observe().images) == {"top", "right_wrist"}


def test_gripper_swap_by_config():
    r = cfg("configs/robots/yam_bimanual.yaml")
    r["gripper"] = "parallel_jaw"
    env = SimEnv(r, make_task(cfg("configs/tasks/sponge_scrub.yaml")))
    env.reset(seed=0)
    env.step({"left": np.r_[np.zeros(6), 0.0], "right": np.r_[np.zeros(6), 1.0]})
    for _ in range(10):
        env.hold()
    s = env.state()
    assert s["left"][-1] < 0.2 and s["right"][-1] > 0.8


@pytest.mark.parametrize("level", ["none", "pose", "full"])
def test_same_seed_same_scene(level):
    a = sponge_env()
    va = a.reset(seed=7, level=level)
    vb = a.reset(seed=7, level=level)
    assert va == vb


def test_pick_place_builds_and_scores():
    env = SimEnv(cfg("configs/robots/yam_bimanual.yaml"), make_task(cfg("configs/tasks/pick_place.yaml")))
    env.reset(seed=0, level="pose")
    for _ in range(5):
        env.hold()
    m = env.task.metrics()
    assert m["progress"] == 0.0 and m["final_distance"] > 0.05
