import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from visual_servo import servo_core as sc

XACRO = os.path.join(os.path.dirname(__file__), '..', '..', 'twolink_gazebo', 'urdf', 'twolinkbot.urdf.xacro')


def urdf_transform(joint, q=0.0):
    origin = joint.find('origin')
    if origin is None:
        origin = ET.Element('origin')
    xyz = [float(v) for v in (origin.get('xyz') or '0 0 0').split()]
    rpy = [float(v) for v in (origin.get('rpy') or '0 0 0').split()]
    t = sc.transform(xyz, rpy)
    if joint.get('type') in ('continuous', 'revolute'):
        t = t @ sc.transform(rpy=(0.0, 0.0, q))
    return t


@pytest.mark.skipif(shutil.which('xacro') is None, reason='xacro not available')
def test_constants_match_urdf():
    out = subprocess.run(['xacro', XACRO], capture_output=True, text=True, check=True).stdout
    joints = {j.get('name'): j for j in ET.fromstring(out).findall('joint')}
    for q in ([0.0, math.pi / 2], [0.7, -1.2], [-1.3, 2.1]):
        expected = (urdf_transform(joints['base_joint']) @ urdf_transform(joints['joint1'], q[0])
                    @ urdf_transform(joints['joint2'], q[1]) @ urdf_transform(joints['camera_joint'])
                    @ urdf_transform(joints['camera_color_joint'])
                    @ urdf_transform(joints['camera_color_optical_joint']))
        assert np.allclose(sc.camera_transform(q, sc.mount_transform()), expected, atol=1e-9)


def test_optical_axis_along_link2_at_start_pose():
    t = sc.camera_transform([0.0, math.pi / 2], sc.mount_transform())
    assert np.allclose(t[:3, 2], [0.0, 1.0, 0.0], atol=0.01)
    assert np.allclose(t[:3, 0], [1.0, 0.0, 0.0], atol=0.01)


def test_virtual_fk_matches_full_fk():
    mount = sc.mount_transform()
    l2, alpha = sc.virtual_link(mount, sc.D_GOAL)
    for q in ([0.0, math.pi / 2], [0.4, -0.8], [-1.0, 2.0]):
        tip = sc.camera_transform(q, mount) @ np.array([0.0, 0.0, sc.D_GOAL, 1.0])
        assert np.allclose(sc.virtual_fk(q, l2, alpha), tip[:2], atol=2e-3)


def test_ik_round_trip():
    servo = sc.VisualServo()
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(500):
        q = rng.uniform(sc.Q_MIN, sc.Q_MAX)
        branch = 1.0 if math.sin(q[1] + servo.alpha) >= 0 else -1.0
        m = sc.virtual_fk(q, servo.l2, servo.alpha)
        r_lo, r_hi = sc.reach_ring(servo.l2, servo.alpha, branch)
        if not r_lo <= math.hypot(m[0], m[1]) <= r_hi:
            continue
        q_ik = sc.ik_branch(m, servo.l2, servo.alpha, branch)
        assert np.allclose(sc.virtual_fk(q_ik, servo.l2, servo.alpha), m, atol=1e-6)
        checked += 1
    assert checked > 300


def test_jacobian_matches_finite_difference():
    servo = sc.VisualServo()
    q = np.array([0.3, 1.1])
    eps = 1e-6
    numeric = np.column_stack([
        (sc.virtual_fk(q + eps * np.eye(2)[i], servo.l2, servo.alpha)
         - sc.virtual_fk(q - eps * np.eye(2)[i], servo.l2, servo.alpha)) / (2 * eps) for i in range(2)])
    assert np.allclose(sc.jacobian(q, servo.l2, servo.alpha), numeric, atol=1e-6)


def test_converges_on_static_marker():
    servo = sc.VisualServo()
    q = np.array([0.0, math.pi / 2])
    target_q = np.array([0.3, 1.2])
    marker = sc.camera_transform(target_q, servo.mount) @ np.array([0.0, 0.0, sc.D_GOAL, 1.0])
    servo.reset(q)
    for _ in range(100):
        p_cam = (np.linalg.inv(sc.camera_transform(q, servo.mount)) @ marker)[:3]
        _, info = servo.step(p_cam, q)
        q = info['q_ref']
    p_cam = (np.linalg.inv(sc.camera_transform(q, servo.mount)) @ marker)[:3]
    e_theta, e_d, _ = sc.marker_errors(p_cam, d_goal=servo.d_goal)
    assert abs(e_theta) < math.radians(1.0)
    assert abs(e_d) < 0.005


def test_motion_is_velocity_and_acceleration_limited():
    servo = sc.VisualServo()
    q = np.array([0.0, math.pi / 2])
    servo.reset(q)
    far = sc.camera_transform([1.2, 0.5], servo.mount) @ np.array([0.0, 0.0, sc.D_GOAL, 1.0])
    dt = 1 / 30
    prev_ref, prev_v = q.copy(), np.zeros(2)
    for _ in range(60):
        p_cam = (np.linalg.inv(sc.camera_transform(q, servo.mount)) @ far)[:3]
        out, info = servo.step(p_cam, q, dt=dt)
        v = (info['q_ref'] - prev_ref) / dt
        assert np.all(np.abs(v) <= sc.V_MAX + 1e-6)
        assert np.all(np.abs(v - prev_v) / dt <= sc.A_MAX * 1.01 + 1e-6)
        assert np.all(np.abs(out - info['q_ref']) <= sc.LEAD_MAX + 1e-9)
        prev_ref, prev_v, q = info['q_ref'], v, info['q_ref']


def test_holds_before_first_marker_and_keeps_going_after_loss():
    servo = sc.VisualServo()
    q = np.array([0.0, math.pi / 2])
    servo.reset(q)
    out, info = servo.step(None, q)
    assert info['held'] and np.allclose(out, q)
    target_q = np.array([0.5, 1.2])
    marker = sc.camera_transform(target_q, servo.mount) @ np.array([0.0, 0.0, sc.D_GOAL, 1.0])
    p_cam = (np.linalg.inv(sc.camera_transform(q, servo.mount)) @ marker)[:3]
    servo.step(p_cam, q, dt=1 / 30)
    for _ in range(90):
        out, info = servo.step(None, q, dt=1 / 30)
    assert info['lost'] and not info['held']
    assert np.allclose(info['q_ref'], target_q, atol=0.02)


def test_unreachable_marker_stays_within_limits():
    servo = sc.VisualServo()
    q = np.array([0.0, math.pi / 2])
    servo.reset(q)
    for p_cam in ((0.0, 0.0, 0.05), (0.0, 0.0, 2.0), (0.5, 0.0, 0.1)):
        for _ in range(50):
            q_cmd, info = servo.step(p_cam, q)
            assert np.all(q_cmd >= sc.Q_MIN) and np.all(q_cmd <= sc.Q_MAX)


def test_marker_errors_signs():
    e_theta, e_d, e_psi = sc.marker_errors((0.05, 0.0, 0.20), (1.0, 0.0, 0.0, 0.0))
    assert e_theta > 0 and abs(e_d) < 1e-12 and abs(e_psi) < 1e-9


def run_static(servo, q, marker, steps=150):
    servo.reset(q)
    modes = []
    for _ in range(steps):
        p_cam = (np.linalg.inv(sc.camera_transform(q, servo.mount)) @ marker)[:3]
        _, info = servo.step(p_cam, q)
        q = info['q_ref']
        modes.append(info['mode'])
    p_cam = (np.linalg.inv(sc.camera_transform(q, servo.mount)) @ marker)[:3]
    return q, sc.marker_errors(p_cam, d_goal=servo.d_goal), modes


def marker_ahead(q, dist, lateral=0.0, mount=None):
    mount = sc.mount_transform() if mount is None else mount
    return sc.camera_transform(q, mount) @ np.array([lateral, 0.0, dist, 1.0])


def test_too_close_marker_keeps_centring_and_stays_off_limits():
    servo = sc.VisualServo()
    q0 = np.array([0.0, math.pi / 2])
    for dist in (0.12, 0.08):
        q, (e_theta, e_d, _), modes = run_static(servo, q0, marker_ahead(q0, dist, 0.02))
        assert abs(e_theta) < math.radians(1.5)
        assert np.all(np.abs(q) < np.array([1.55, 2.45]))


def test_far_marker_keeps_centring():
    servo = sc.VisualServo()
    q0 = np.array([0.0, math.pi / 2])
    q, (e_theta, e_d, _), modes = run_static(servo, q0, marker_ahead(q0, 0.70, 0.10))
    assert abs(e_theta) < math.radians(1.5)
    assert 'distance_limited' in modes


def test_tracker_follows_a_moving_marker_and_predicts_ahead():
    tr = sc.MarkerTracker()
    v = np.array([0.1, 0.0])
    for k in range(60):
        t = k / 30
        tr.update(np.array([0.3, 0.1]) + v * t, t)
    assert np.allclose(tr.v, v, atol=0.01)
    assert np.allclose(tr.predict(t, 0.0), np.array([0.3, 0.1]) + v * t, atol=0.003)
    assert tr.predict(t, 0.25)[0] > tr.predict(t, 0.0)[0] + 0.015


def test_joint1_limit_centres_with_joint2():
    servo = sc.VisualServo()
    q0 = np.array([2.85, 0.6])
    marker = marker_ahead(np.array([3.3, 0.4]), servo.d_goal)
    q, (e_theta, _, _), modes = run_static(servo, q0, marker)
    assert 'joint_limited' in modes
    assert abs(q[0] - sc.Q_MAX[0]) <= 0.01
    assert abs(e_theta) < math.radians(1.5)


def test_elbow_branch_never_flips():
    servo = sc.VisualServo()
    q = np.array([0.0, math.pi / 2])
    servo.reset(q)
    rng = np.random.default_rng(1)
    for _ in range(400):
        p_cam = (rng.uniform(-0.12, 0.12), rng.uniform(-0.02, 0.02), rng.uniform(0.06, 0.8))
        out, info = servo.step(p_cam, q)
        q = info['q_ref']
        assert math.sin(q[1] + servo.alpha) > 0 and math.sin(out[1] + servo.alpha) > 0
        assert np.all(q >= sc.Q_MIN) and np.all(q <= sc.Q_MAX)


def test_no_wrap_around_swing_past_the_limit():
    servo = sc.VisualServo()
    q = np.array([2.9, 1.0])
    servo.reset(q)
    marker = marker_ahead(np.array([3.4, 0.9]), servo.d_goal)
    for _ in range(80):
        p_cam = (np.linalg.inv(sc.camera_transform(q, servo.mount)) @ marker)[:3]
        q_new, _ = servo.step(p_cam, q)
        assert np.all(np.abs(q_new - q) <= sc.V_MAX * 0.1 + sc.LEAD_MAX + 1e-9)
        assert q_new[0] > 2.5
        q = q_new


def test_far_marker_with_stretched_arm_is_tracked_not_held():
    servo = sc.VisualServo()
    q = np.array([1.67892355, 0.24237344])
    servo.reset(q)
    marker = np.array([-0.17838204, 0.45846629, 0.18, 1.0])
    p_cam = (np.linalg.inv(sc.camera_transform(q, servo.mount)) @ marker)[:3]
    _, info = servo.step(p_cam, q)
    assert not info['held'] and info['mode'] == 'distance_limited'
    assert 0.2 < info['d_target'] < 0.3


def test_never_holds_a_visible_marker_in_front():
    servo = sc.VisualServo()
    rng = np.random.default_rng(3)
    for _ in range(300):
        q = rng.uniform(sc.Q_MIN * 0.9, sc.Q_MAX * 0.9)
        if math.sin(q[1] + servo.alpha) < 0.1:
            continue
        servo.reset(q)
        p_cam = (rng.uniform(-0.1, 0.1), 0.0, rng.uniform(0.1, 0.6))
        m = sc.camera_transform(q, servo.mount) @ np.array([p_cam[0], 0.0, p_cam[2], 1.0])
        if math.hypot(m[0], m[1]) > servo.ring(sc.D_RANGE[1])[1] or math.hypot(m[0], m[1]) < servo.ring(sc.D_RANGE[0])[0]:
            continue
        _, info = servo.step(p_cam, q)
        assert not info['held'], (q, p_cam)
