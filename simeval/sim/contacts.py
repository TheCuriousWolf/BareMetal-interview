"""Contact queries used by task scoring. All work on body ids and read ground-truth simulator state."""

from __future__ import annotations

import mujoco
import numpy as np


def contact_force(model: mujoco.MjModel, data: mujoco.MjData, i: int) -> np.ndarray:
    f = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, f)
    return f[:3]  # normal, tangent1, tangent2 in the contact frame


def contacts_between(model, data, bodies_a: set[int], bodies_b: set[int]):
    """Yield (contact index, contact, normal force) for every contact between the two body sets."""
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if (b1 in bodies_a and b2 in bodies_b) or (b1 in bodies_b and b2 in bodies_a):
            yield i, c, abs(contact_force(model, data, i)[0])


def total_normal_force(model, data, bodies_a: set[int], bodies_b: set[int]) -> float:
    return float(sum(f for _, _, f in contacts_between(model, data, bodies_a, bodies_b)))


def is_grasped(model, data, finger_ids: set[int], object_ids: set[int], min_force: float = 0.1) -> bool:
    """Held when every finger touches the object with at least min_force."""
    touching = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        for f, o in ((b1, b2), (b2, b1)):
            if f in finger_ids and o in object_ids and abs(contact_force(model, data, i)[0]) >= min_force:
                touching.add(f)
    return bool(finger_ids) and touching == finger_ids


def body_subtree(model, root_id: int) -> set[int]:
    return {i for i in range(model.nbody) if _is_descendant(model, i, root_id)}


def _is_descendant(model, i: int, root: int) -> bool:
    while i != 0:
        if i == root:
            return True
        i = model.body_parentid[i]
    return root == 0
