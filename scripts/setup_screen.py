"""Evaluation setup screen: build a scene from interactables and define metrics, stages and end conditions.

    python scripts/setup_screen.py            # then open http://localhost:8765

Reads and writes the same YAML files the runner uses (scenes/*.yaml, configs/tasks/*.yaml). "Preview" builds the
scene with the chosen robot and shows the station camera.
"""

import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from simeval.paths import CONFIGS, SCENES  # noqa: E402

PAGE = ROOT / "scripts" / "setup_screen.html"

# What the screen offers. Kept here (not in the page) so new types show up by editing Python only.
OBJECT_TYPES = {
    "surface": {"pos": [0, -0.12, 0.73], "half_size": [0.65, 0.55, 0.03], "friction": 1.0},
    "object": {"surface": "", "shape": "box", "half_size": [0.02, 0.02, 0.02], "mass": 0.1, "xy": [0, 0],
               "yaw_deg": 0, "rgba": [0.85, 0.2, 0.2, 1]},
    "sponge": {"surface": "", "half_size": [0.045, 0.03, 0.0175], "mass": 0.025, "xy": [-0.13, 0.07], "yaw_deg": 0,
               "rgba": [0.96, 0.82, 0.25, 1]},
    "spill": {"surface": "", "erased_by": "", "center_xy": [0.06, -0.02], "radius": 0.04, "min_press_force": 0.5,
              "scrub_distance": 0.08, "require_grasp": True},
    "region": {"surface": "", "center_xy": [0.1, -0.02], "radius": 0.05},
}
METRIC_TYPES = {
    "distance": ["a", "b"], "contact": ["a", "b"], "contact_time": ["a", "b"], "force": ["a", "b"],
    "grasped": ["object"], "grasp_time": ["object"], "height": ["object"], "moved": ["object"],
    "in_region": ["object", "region"], "value": ["object", "key"],
}
VALUE_KEYS = {"spill": ["cleaned", "remaining"]}


def robot_refs(robot_file: str) -> list[str]:
    cfg = yaml.safe_load((ROOT / robot_file).read_text())
    refs = ["robot.any.eef", "robot.any.gripper", "robot.all"]
    for m in cfg["mounts"]:
        refs += [f"robot.{m}.eef", f"robot.{m}.gripper", f"robot.{m}"]
    return refs


def preview_png(task_cfg: dict, scene: dict, robot_file: str, seed: int, level: str) -> bytes:
    from PIL import Image

    from simeval.sim.sim_env import SimEnv
    from simeval.tasks.task import Task

    tmp = SCENES / "_preview.yaml"
    tmp.write_text(yaml.safe_dump(scene, sort_keys=False))
    try:
        cfg = {**task_cfg, "scene": tmp.name, "scene_overrides": {}}
        env = SimEnv(yaml.safe_load((ROOT / robot_file).read_text()), Task(cfg))
        env.reset(seed=seed, level=level)
        img = env.observe().images.get("top")
        env.close()
    finally:
        tmp.unlink(missing_ok=True)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            return self._send(PAGE.read_bytes(), "text/html")
        if url.path == "/api/catalog":
            return self._send({
                "object_types": OBJECT_TYPES, "metric_types": METRIC_TYPES, "value_keys": VALUE_KEYS,
                "tasks": sorted(p.name for p in (CONFIGS / "tasks").glob("*.yaml")),
                "robots": sorted(f"configs/robots/{p.name}" for p in (CONFIGS / "robots").glob("*.yaml")),
            })
        if url.path == "/api/robot_refs":
            return self._send(robot_refs(q["robot"]))
        if url.path == "/api/task":
            task = yaml.safe_load((CONFIGS / "tasks" / q["name"]).read_text())
            scene = yaml.safe_load((SCENES / task["scene"]).read_text())
            return self._send({"task": task, "scene": scene})
        self._send({"error": "not found"}, code=404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            if self.path == "/api/save":
                task, scene = body["task"], body["scene"]
                name = Path(body["task_file"]).name
                (SCENES / Path(task["scene"]).name).write_text(yaml.safe_dump(scene, sort_keys=False))
                (CONFIGS / "tasks" / name).write_text(yaml.safe_dump(task, sort_keys=False))
                return self._send({"saved": [f"configs/tasks/{name}", f"scenes/{task['scene']}"]})
            if self.path == "/api/preview":
                png = preview_png(body["task"], body["scene"], body["robot"], int(body.get("seed", 0)),
                                  body.get("level", "none"))
                return self._send(png, "image/png")
        except Exception as e:  # show the problem in the page instead of dropping the connection
            return self._send({"error": f"{type(e).__name__}: {e}"}, code=400)
        self._send({"error": "not found"}, code=404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"Setup screen on http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
