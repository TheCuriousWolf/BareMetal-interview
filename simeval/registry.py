"""Name -> class lookup for tasks and policies.

Built-ins are listed here; anything else can be referenced from config as "package.module:ClassName", so a new
provider or task in another repo plugs in without editing this file.
"""

from __future__ import annotations

import importlib

TASKS = {
    "sponge_scrub": "simeval.tasks.sponge_scrub:SpongeScrubTask",
    "pick_place": "simeval.tasks.pick_place:PickPlaceTask",
}
POLICIES = {
    "dreamlabs": "simeval.policies.dreamlabs:DreamLabsPolicy",
    "hold": "simeval.policies.baselines:HoldPolicy",
    "random": "simeval.policies.baselines:RandomPolicy",
}


def resolve(name: str, table: dict[str, str]):
    target = table.get(name, name)
    if ":" not in target:
        raise KeyError(f"Unknown '{name}'. Known: {sorted(table)}; or use 'module:Class'.")
    module, cls = target.split(":")
    return getattr(importlib.import_module(module), cls)


def make_task(cfg: dict):
    return resolve(cfg["task"], TASKS)(cfg)


def make_policy(cfg: dict):
    return resolve(cfg["policy"], POLICIES)(cfg)
