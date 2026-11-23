"""Writes gripper.xml: the stock YAM linear (parallel) gripper as a standalone MJCF model.

Geometry, finger slides and contact tuning follow the Dream Labs YAM MuJoCo example (Apache-2.0, meshes MIT, I2RT).
Visual meshes are the stock STLs; contacts use the convex decomposition in collision_parts.npz.
Run once after changing meshes or parameters: python robot_models/grippers/yam_linear/build_gripper.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STROKE = 0.0475  # metres of travel per finger
# Mesh origin offsets inside each body (from the YAM model description).
GEOM_OFFSETS = {
    "gripper.stl": ("-.014 -.0463995 .0731", "1 0 0 0"),
    "tip_left.stl": (".129783 .00999321 -.0914614", ".499998 .5 .500002 .5"),
    "tip_right.stl": ("-.0379932 .129783 .00133753", ".707105 -.707108 0 0"),
}


def num(values):
    return " ".join(f"{float(v):.6g}" for v in values)


def main():
    parts = {}
    with np.load(HERE / "collision_parts.npz") as packed:
        for key in packed.files:
            source, index = key.rsplit("/", 1)
            parts.setdefault(source, {})[int(index)] = packed[key]

    root = ET.Element("mujoco", model="yam_linear_gripper")
    ET.SubElement(root, "compiler", angle="radian", meshdir="meshes")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    contact = ET.SubElement(root, "contact")
    actuator = ET.SubElement(root, "actuator")
    ET.SubElement(root, "equality")

    def add_mesh_geoms(body, file, finger):
        name = Path(file).stem
        pos, quat = GEOM_OFFSETS[file]
        ET.SubElement(asset, "mesh", name=name, file=file, smoothnormal="true")
        ET.SubElement(body, "geom", type="mesh", mesh=name, pos=pos, quat=quat, rgba=".065 .07 .08 1", contype="0",
                      conaffinity="0", mass="0", group="1")
        for i in sorted(parts[file]):
            part = f"{name}_part{i}"
            ET.SubElement(asset, "mesh", name=part, vertex=num(parts[file][i].ravel()))
            attrs = dict(name=f"{name}_contact_{i}", type="mesh", mesh=part, pos=pos, quat=quat, rgba="0 0 0 0",
                         mass="0", group="3")
            if finger:
                attrs.update(friction="6 .08 .008", condim="6", priority="1", solref=".012 1", solimp=".95 .99 .001")
            ET.SubElement(body, "geom", **attrs)

    # Root body frame is the gripper frame of the YAM model; the robot config places it on the arm flange.
    gripper = ET.SubElement(world, "body", name="gripper", gravcomp="1")
    add_mesh_geoms(gripper, "gripper.stl", finger=False)
    tcp = ET.SubElement(gripper, "body", name="grasp_site", pos="0 0 -.1347", quat="0 1 0 0", gravcomp="1")
    ET.SubElement(tcp, "inertial", pos="0 0 -.09", mass=".1", diaginertia=".00003 .00004 .00005")
    ET.SubElement(tcp, "site", name="tcp", size=".004", rgba="0 1 0 0")
    for name, pos, quat in [
        ("tip_left", "-.0238981 .0450619 -.0545599", ".499998 -.5 -.5 -.500002"),
        ("tip_right", ".0238981 -.0450619 -.0545599", ".707105 .707108 0 0"),
    ]:
        tip = ET.SubElement(gripper, "body", name=name, pos=pos, quat=quat, gravcomp="1")
        joint = f"finger_{name.split('_')[1]}"
        ET.SubElement(tip, "joint", name=joint, type="slide", axis="0 0 -1", range=f"0 {STROKE}", damping="1",
                      armature=".02")
        ET.SubElement(tip, "inertial", pos="0 0 0", mass=".071", diaginertia=".00005 .00005 .00005")
        add_mesh_geoms(tip, f"{name}.stl", finger=True)
        ET.SubElement(actuator, "position", name=joint, joint=joint, kp="1000", kv="30", ctrlrange=f"0 {STROKE}",
                      forcerange="-22 22")
    # Closed CAD finger tips overlap slightly; the slide limits model the stop.
    ET.SubElement(contact, "exclude", body1="tip_left", body2="tip_right")
    ET.indent(root)
    (HERE / "gripper.xml").write_text(ET.tostring(root, encoding="unicode"))
    print("wrote", HERE / "gripper.xml")


if __name__ == "__main__":
    main()
