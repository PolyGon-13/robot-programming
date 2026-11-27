import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src", "visual_servo", "visual_servo"))
import servo_core as sc


def mat_to_quat(r):
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return w, x, y, z


def fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def build(cam_center_y, cam_bottom_z, kp, kv):
    optical = sc.mount_transform(cam_center_y, cam_bottom_z)
    camera_link = sc.transform((sc.CAM_BACK_X + sc.CAM_DEPTH - sc.CAM_ORIGIN_BEHIND_FRONT,
                                cam_center_y + sc.CAM_ORIGIN_FROM_CENTER_Y, cam_bottom_z + sc.CAM_HEIGHT / 2))
    box_center = np.array([sc.CAM_BACK_X + sc.CAM_DEPTH / 2, cam_center_y, cam_bottom_z + sc.CAM_HEIGHT / 2])
    r_opt = optical[:3, :3]
    xy_axes = np.concatenate((r_opt[:, 0], -r_opt[:, 1]))
    fovy = math.degrees(2 * math.atan(242.520 / 606.322))
    return f"""<mujoco model="twolinkbot">
  <compiler angle="radian" meshdir="../assets/twolink/meshes"/>
  <option timestep="0.00166666667" gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <asset>
    <mesh name="base" file="base.STL"/>
    <mesh name="link1" file="link1.STL"/>
    <mesh name="link2" file="link2.STL"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.25 0.25 0.25" rgb2="0.3 0.3 0.3"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
    <material name="grey" rgba="0.6 0.6 0.6 1"/>
    <material name="dark" rgba="0.15 0.15 0.15 1"/>
  </asset>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="2 2 0.01" material="grid"/>
    <body name="base_link">
      <inertial pos="-0.0455 -0.0455 0.0455" mass="0.1" diaginertia="0.000131508 0.000131508 0.000125"/>
      <geom type="mesh" mesh="base" material="grey"/>
      <body name="link1" pos="0 0 0.128">
        <inertial pos="-0.0715 0 0.03" mass="0.5" diaginertia="0.00027075 0.00180204 0.00180204"/>
        <joint name="joint1" type="hinge" axis="0 0 1" range="{sc.Q_MIN[0]} {sc.Q_MAX[0]}" limited="true" damping="0.01"/>
        <geom type="mesh" mesh="link1" material="grey"/>
        <body name="link2" pos="0.143 0 0.052">
          <inertial pos="-0.051 0 0.003" mass="0.2" diaginertia="3.9e-05 0.0003756 0.0004134"/>
          <joint name="joint2" type="hinge" axis="0 0 1" range="{sc.Q_MIN[1]} {sc.Q_MAX[1]}" limited="true" damping="0.01"/>
          <geom type="mesh" mesh="link2" material="grey"/>
          <geom name="d435" type="box" size="{fmt([sc.CAM_DEPTH / 2, sc.CAM_WIDTH / 2, sc.CAM_HEIGHT / 2])}" pos="{fmt(box_center)}" material="dark"/>
          <site name="color_optical" pos="{fmt(optical[:3, 3])}" quat="{fmt(mat_to_quat(r_opt))}" size="0.004" rgba="0 1 0 1"/>
          <camera name="color" pos="{fmt(optical[:3, 3])}" xyaxes="{fmt(xy_axes)}" fovy="{fovy:.4f}"/>
        </body>
      </body>
    </body>
    <body name="marker" mocap="true" pos="0.5 0 0.2">
      <geom type="box" size="0.03 0.03 0.001" rgba="0.95 0.95 0.95 1"/>
      <geom type="box" size="0.02 0.02 0.0012" rgba="0.05 0.05 0.05 1"/>
      <site name="marker_frame" size="0.003" rgba="1 0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="joint1" joint="joint1" kp="{kp[0]}" kv="{kv[0]}" ctrlrange="{sc.Q_MIN[0]} {sc.Q_MAX[0]}" ctrllimited="true"/>
    <position name="joint2" joint="joint2" kp="{kp[1]}" kv="{kv[1]}" ctrlrange="{sc.Q_MIN[1]} {sc.Q_MAX[1]}" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cam_center_y", type=float, default=sc.CAM_CENTER_Y)
    p.add_argument("--cam_bottom_z", type=float, default=sc.CAM_BOTTOM_Z)
    p.add_argument("--out", type=str, default=os.path.join(HERE, "twolink.xml"))
    a = p.parse_args()
    xml = build(a.cam_center_y, a.cam_bottom_z, kp=(4.0, 1.0), kv=(0.4, 0.08))
    with open(a.out, "w") as f:
        f.write(xml)
    import mujoco

    m = mujoco.MjModel.from_xml_path(a.out)
    d = mujoco.MjData(m)
    worst = 0.0
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(sc.Q_MIN * 0.9, sc.Q_MAX * 0.9)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        sid = m.site("color_optical").id
        t_ref = sc.camera_transform(q, sc.mount_transform(a.cam_center_y, a.cam_bottom_z))
        worst = max(worst, np.abs(d.site_xpos[sid] - t_ref[:3, 3]).max(), np.abs(d.site_xmat[sid].reshape(3, 3) - t_ref[:3, :3]).max())
    print("MJCF", a.out, "nq", m.nq, "nu", m.nu, "max |site - servo_core camera_transform|", worst)
