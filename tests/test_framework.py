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
    t["scene_overrides"] = {"objects": {"stain": {"require_grasp": require_grasp}}}
    env = SimEnv(cfg(robot), make_task(t))
    env.reset(seed=0, level="none")
    return env


def scrub(env, press_newtons, strokes=6):
    """Drive the sponge back and forth across the stain, optionally pushing it down."""
    m, d, task = env.model, env.data, env.task
    sponge, stain = task.objects["sponge"], task.objects["stain"]
    jnt = m.joint("sponge_free")
    adr, dof = jnt.qposadr[0], jnt.dofadr[0]
    cx, cy = stain.cells.mean(axis=0)
    z = stain.top + sponge.size[2]
    for k in range(strokes * 20):
        x = cx + 0.05 * np.sin(k / 20 * np.pi)
        d.qpos[adr : adr + 3] = [x, cy, z]
        d.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        d.qvel[dof : dof + 6] = 0
        d.xfrc_applied[m.body("sponge").id] = [0, 0, -press_newtons, 0, 0, 0]
        env.hold()
    return task.metrics()


def test_scrubbing_needs_pressure_and_motion():
    pressed = scrub(sponge_env(require_grasp=False), press_newtons=5.0)
    light = scrub(sponge_env(require_grasp=False), press_newtons=0.0)
    assert pressed["stain_remaining"] < 0.8 and pressed["stage_times"]["stain_contact"] is not None
    assert light["stain_remaining"] == 1.0  # resting weight alone is below min_press_force


def test_loose_sponge_does_not_count_when_grasp_required():
    assert scrub(sponge_env(require_grasp=True), press_newtons=5.0)["stain_remaining"] == 1.0


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
    assert m["progress"] == 0.0 and m["cube_to_target"] > 0.05 and m["cube_in_target"] == 0


def test_metrics_follow_the_robot():
    """Contact time and distance metrics respond when the end effector is driven onto the sponge."""
    env = sponge_env()
    m, d = env.model, env.data
    start = env.task.metrics()["eef_to_sponge"]
    # Teleport the sponge between the right gripper's fingers and close them.
    tcp = d.site("right_tcp").xpos.copy()
    jnt = m.joint("sponge_free")
    d.qpos[jnt.qposadr[0]: jnt.qposadr[0] + 3] = tcp
    for _ in range(20):
        env.step({"left": env.state()["left"], "right": np.r_[env.state()["right"][:6], 0.0]})
    out = env.task.metrics()
    assert out["eef_to_sponge"] < start and out["eef_to_sponge.min"] < 0.06
    assert out["gripper_sponge_time"] > 0


def test_fk_ik_round_trip():
    env = sponge_env()
    robot, d = env.robot, env.data
    pos, _ = robot.fk(d, "right")
    goal = pos + np.array([0.0, -0.10, -0.08])
    q = robot.ik(d, "right", goal)
    for _ in range(60):
        env.step({"left": env.state()["left"], "right": np.r_[q, 1.0]})
    reached, _ = robot.fk(d, "right")
    assert np.linalg.norm(reached - goal) < 0.02
    assert set(env.observe(render=False).sensors) >= {"right.eef_pos", "right.joint_vel"}
