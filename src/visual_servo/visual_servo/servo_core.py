import math

import numpy as np

L1 = 0.143
JOINT1_Z = 0.128
JOINT2_Z = 0.052
D_GOAL = 0.20
Q_MIN = np.array([-3.0, -3.0])
Q_MAX = np.array([3.0, 3.0])
V_MAX = 2.5
A_MAX = 15.0
OMEGA = 14.0
LAG_TAU = 0.15
LEAD_MAX = 0.3
LEAD_TIME = 0.25
COAST_TIME = 1.0
COAST_TAU = 0.4
TRACK_TAU = 0.12
RANGE_TAU = 0.5
TARGET_V_MAX = 0.6
DEADBAND = 0.003
BRANCH_MARGIN = 0.05
D_RANGE = (0.12, 0.80)

CAM_BACK_X = 0.088
CAM_CENTER_Y = 0.0
CAM_BOTTOM_Z = 0.016
CAM_WIDTH = 0.090
CAM_HEIGHT = 0.025
CAM_DEPTH = 0.02505
CAM_ORIGIN_BEHIND_FRONT = 0.0043
CAM_ORIGIN_FROM_CENTER_Y = 0.0175
COLOR_XYZ = (-0.00022, 0.01483, 0.00014)
COLOR_RPY = (0.0084, -0.0010, 0.0045)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    t = np.eye(4)
    t[:3, :3] = rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])
    t[:3, 3] = xyz
    return t


def quat_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mount_transform(cam_center_y=CAM_CENTER_Y, cam_bottom_z=CAM_BOTTOM_Z):
    camera_link = transform((
        CAM_BACK_X + CAM_DEPTH - CAM_ORIGIN_BEHIND_FRONT,
        cam_center_y + CAM_ORIGIN_FROM_CENTER_Y,
        cam_bottom_z + CAM_HEIGHT / 2))
    color = transform(COLOR_XYZ, COLOR_RPY)
    optical = transform(rpy=(-math.pi / 2, 0.0, -math.pi / 2))
    return camera_link @ color @ optical


def link2_transform(q):
    return (transform((0.0, 0.0, JOINT1_Z), (0.0, 0.0, q[0]))
            @ transform((L1, 0.0, JOINT2_Z), (0.0, 0.0, q[1])))


def camera_transform(q, mount):
    return link2_transform(q) @ mount


def virtual_link(mount, d_goal):
    tip = mount[:3, 3] + d_goal * mount[:3, 2]
    return math.hypot(tip[0], tip[1]), math.atan2(tip[1], tip[0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def virtual_fk(q, l2, alpha):
    a1, a12 = q[0], q[0] + q[1] + alpha
    return np.array([L1 * math.cos(a1) + l2 * math.cos(a12), L1 * math.sin(a1) + l2 * math.sin(a12)])


def jacobian(q, l2, alpha):
    a1, a12 = q[0], q[0] + q[1] + alpha
    return np.array([
        [-L1 * math.sin(a1) - l2 * math.sin(a12), -l2 * math.sin(a12)],
        [L1 * math.cos(a1) + l2 * math.cos(a12), l2 * math.cos(a12)]])


def reach_ring(l2, alpha, branch, q_min=Q_MIN, q_max=Q_MAX):
    if branch > 0:
        lo, hi = BRANCH_MARGIN, min(q_max[1] + alpha, math.pi)
    else:
        lo, hi = max(q_min[1] + alpha, -math.pi), -BRANCH_MARGIN
    radius = [math.sqrt(L1 * L1 + l2 * l2 + 2 * L1 * l2 * math.cos(a)) for a in (lo, hi)]
    return min(radius), max(radius)


def ik_branch(target_xy, l2, alpha, branch):
    mx, my = float(target_xy[0]), float(target_xy[1])
    r = math.hypot(mx, my)
    c2 = min(1.0, max(-1.0, (r * r - L1 * L1 - l2 * l2) / (2 * L1 * l2)))
    q2p = branch * math.acos(c2)
    q1 = math.atan2(my, mx) - math.atan2(l2 * math.sin(q2p), L1 + l2 * math.cos(q2p))
    return np.array([wrap(q1), wrap(q2p - alpha)])


def inverse_kinematics(target_xy, l2, alpha, q_ref, q_min=Q_MIN, q_max=Q_MAX):
    branch = 1.0 if math.sin(q_ref[1] + alpha) >= 0 else -1.0
    r_lo, r_hi = reach_ring(l2, alpha, branch, q_min, q_max)
    r = math.hypot(target_xy[0], target_xy[1])
    reachable = r_lo <= r <= r_hi
    q = ik_branch(target_xy, l2, alpha, branch)
    inside = bool(np.all(q >= q_min) and np.all(q <= q_max))
    return np.clip(q, q_min, q_max), reachable, inside


def center_with_q2(target_xy, q1, mount):
    v = np.asarray(target_xy, dtype=float) - L1 * np.array([math.cos(q1), math.sin(q1)])
    o = mount[:2, 3]
    z = mount[:2, 2] / np.linalg.norm(mount[:2, 2])
    b = float(o @ z)
    disc = b * b - float(o @ o) + float(v @ v)
    if disc < 0:
        return None
    dist = -b + math.sqrt(disc)
    if dist <= 0:
        return None
    tip = o + dist * z
    return wrap(math.atan2(v[1], v[0]) - math.atan2(tip[1], tip[0]) - q1)


def marker_errors(p_cam, quat_cam=None, d_goal=D_GOAL):
    x, y, z = p_cam
    e_theta = math.atan2(x, z)
    e_d = z - d_goal
    e_psi = float('nan')
    if quat_cam is not None:
        n = quat_to_matrix(*quat_cam)[:, 2]
        e_psi = math.atan2(n[0], -n[2])
    return e_theta, e_d, e_psi


class MarkerTracker:
    def __init__(self, tau=TRACK_TAU, range_tau=RANGE_TAU, v_max=TARGET_V_MAX, coast_tau=COAST_TAU, reset_gap=1.5):
        self.tau = tau
        self.range_tau = range_tau
        self.v_max = v_max
        self.coast_tau = coast_tau
        self.reset_gap = reset_gap
        self.p = None
        self.v = np.zeros(2)
        self.t = None

    def reset(self):
        self.p = None
        self.v = np.zeros(2)
        self.t = None

    def update(self, m_xy, t, view_dir=None):
        m_xy = np.asarray(m_xy, dtype=float)
        if self.p is None or t - self.t > self.reset_gap:
            self.p, self.v, self.t = m_xy.copy(), np.zeros(2), t
            return
        dt = t - self.t
        if dt <= 0:
            return
        if dt > 0.15:
            self.p, self.v = self.predict(t, 0.0), self.v * math.exp(-dt / self.coast_tau)
        else:
            self.p = self.p + self.v * dt
        r = m_xy - self.p
        parts = [(r, self.tau)]
        if view_dir is not None:
            u = np.asarray(view_dir, dtype=float)
            u = u / max(np.linalg.norm(u), 1e-9)
            along = float(r @ u) * u
            parts = [(r - along, self.tau), (along, self.range_tau)]
        for part, tau in parts:
            theta = math.exp(-dt / tau)
            self.p = self.p + (1 - theta * theta) * part
            self.v = self.v + (1 - theta) ** 2 / dt * part
        speed = float(np.linalg.norm(self.v))
        if speed > self.v_max:
            self.v *= self.v_max / speed
        self.t = t

    def age(self, t):
        return float('inf') if self.t is None else t - self.t

    def predict(self, t, lead):
        dt = max(0.0, t - self.t + lead)
        return self.p + self.v * self.coast_tau * (1 - math.exp(-dt / self.coast_tau))


class VisualServo:
    CONTROL_RATE = 30.0

    def __init__(self, d_goal=D_GOAL, v_max=V_MAX, a_max=A_MAX, omega=OMEGA, lag_tau=LAG_TAU,
                 lead_time=LEAD_TIME, coast_time=COAST_TIME, deadband=DEADBAND,
                 q_min=Q_MIN, q_max=Q_MAX, mount=None):
        self.d_goal = d_goal
        self.v_max = v_max
        self.a_max = a_max
        self.omega = omega
        self.lag_tau = lag_tau
        self.lead_time = lead_time
        self.coast_time = coast_time
        self.deadband = deadband
        self.q_min = np.asarray(q_min, dtype=float)
        self.q_max = np.asarray(q_max, dtype=float)
        self.mount = mount_transform() if mount is None else np.asarray(mount, dtype=float)
        self.l2, self.alpha = virtual_link(self.mount, d_goal)
        self.cam_z = float(link2_transform((0.0, 0.0))[2, 3] + self.mount[2, 3])
        self.tracker = MarkerTracker()
        self.q_cmd = None
        self.qd = np.zeros(2)
        self.branch = 1.0
        self.t = 0.0

    def reset(self, q):
        self.q_cmd = np.clip(np.asarray(q, dtype=float), self.q_min, self.q_max)
        self.qd = np.zeros(2)
        self.branch = 1.0 if math.sin(self.q_cmd[1] + self.alpha) >= 0 else -1.0
        self.tracker.reset()
        self.t = 0.0

    def marker_in_base(self, p_cam, q):
        p = camera_transform(q, self.mount) @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
        return p[:3]

    def ring(self, d):
        l2, alpha = virtual_link(self.mount, d)
        return reach_ring(l2, alpha, self.branch, self.q_min, self.q_max)

    def feasible_distance(self, r, d):
        r_lo, r_hi = self.ring(d)
        if r_lo <= r <= r_hi:
            return d, True
        if r < r_lo:
            lo, hi = D_RANGE[0], d
            if r < self.ring(lo)[0]:
                return None, False
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if r >= self.ring(mid)[0]:
                    lo = mid
                else:
                    hi = mid
            return lo, False
        lo, hi = d, D_RANGE[1]
        if r > self.ring(hi)[1]:
            return None, False
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if r <= self.ring(mid)[1]:
                hi = mid
            else:
                lo = mid
        return hi, False

    def joint_target(self, m_xy):
        m = np.array([m_xy[0], m_xy[1], self.cam_z, 1.0])
        d_target = self.d_goal
        d_target, reachable = self.feasible_distance(math.hypot(m[0], m[1]), d_target)
        if d_target is None:
            return None, 'hold', False, float('nan')
        l2, alpha = virtual_link(self.mount, d_target)
        q_ik = ik_branch(m[:2], l2, alpha, self.branch)
        q_ik = self.q_cmd + np.array([wrap(q_ik[0] - self.q_cmd[0]), wrap(q_ik[1] - self.q_cmd[1])])
        mode = 'track' if reachable else 'distance_limited'
        if not self.q_min[0] <= q_ik[0] <= self.q_max[0]:
            q1 = float(np.clip(q_ik[0], self.q_min[0], self.q_max[0]))
            q2 = center_with_q2(m[:2], q1, self.mount)
            if q2 is None:
                return None, 'hold', reachable, d_target
            q_ik = np.array([q1, self.q_cmd[1] + wrap(q2 - self.q_cmd[1])])
            mode = 'joint_limited'
        return q_ik, mode, reachable, d_target

    def advance(self, q_target, dt):
        n = max(1, int(math.ceil(dt / 0.02)))
        for _ in range(n):
            self.advance_once(q_target, dt / n)

    def advance_once(self, q_target, dt):
        for j in range(2):
            e = (q_target[j] - self.q_cmd[j]) if q_target is not None else 0.0
            if abs(e) < self.deadband and abs(self.qd[j]) < 0.05:
                e = 0.0
            a = self.omega * self.omega * e - 2 * self.omega * self.qd[j]
            a = min(self.a_max, max(-self.a_max, a))
            v = self.qd[j] + a * dt
            v = min(self.v_max, max(-self.v_max, v))
            brake = math.sqrt(2 * self.a_max * abs(e))
            if e != 0.0 and v * math.copysign(1.0, e) > brake:
                v = math.copysign(brake, e)
            q = self.q_cmd[j] + v * dt
            if q <= self.q_min[j] or q >= self.q_max[j]:
                q = min(self.q_max[j], max(self.q_min[j], q))
                v = 0.0
            if j == 1 and self.branch * math.sin(q + self.alpha) < 0.02 and abs(q + self.alpha) < 1.0:
                q = self.branch * 0.02 - self.alpha
                v = 0.0
            self.q_cmd[j], self.qd[j] = q, v

    def step(self, p_cam, q_meas, quat_cam=None, dt=0.1):
        q_meas = np.asarray(q_meas, dtype=float)
        if self.q_cmd is None:
            self.reset(q_meas)
        self.t += dt
        info = {'held': True, 'mode': 'hold', 'seen': p_cam is not None, 'lost': True, 'reachable': None,
                'inside_limits': None, 'q_ik': None, 'marker_base': None, 'target': None, 'd_target': float('nan'),
                'e_theta': float('nan'), 'e_d': float('nan'), 'e_psi': float('nan')}
        if p_cam is not None:
            info['e_theta'], info['e_d'], info['e_psi'] = marker_errors(p_cam, quat_cam, self.d_goal)
            if p_cam[2] > 0.03:
                m = self.marker_in_base(p_cam, q_meas)
                info['marker_base'] = m
                cam = camera_transform(q_meas, self.mount)[:2, 3]
                self.tracker.update(m[:2], self.t, m[:2] - cam)
        q_target = None
        if self.tracker.p is not None:
            age = self.tracker.age(self.t)
            info['lost'] = age > self.coast_time
            target = self.tracker.predict(self.t, self.lead_time if age <= self.coast_time else 0.0)
            if info['lost']:
                target = self.tracker.predict(self.tracker.t + self.coast_time, 0.0)
            info['target'] = target
            q_target, mode, reachable, d_target = self.joint_target(target)
            if q_target is not None:
                inside = bool(np.all(q_target >= self.q_min) and np.all(q_target <= self.q_max))
                q_target = np.clip(q_target, self.q_min, self.q_max)
                info.update(held=False, mode=mode, reachable=reachable, inside_limits=inside, q_ik=q_target,
                            d_target=d_target)
        self.advance(q_target, dt)
        lead = np.clip(self.lag_tau * self.qd, -LEAD_MAX, LEAD_MAX)
        q_out = np.clip(self.q_cmd + lead, self.q_min, self.q_max)
        if self.branch * math.sin(q_out[1] + self.alpha) < 0.02 and abs(q_out[1] + self.alpha) < 1.0:
            q_out[1] = self.q_cmd[1]
        info['q_ref'] = self.q_cmd.copy()
        return q_out, info


_default = None


def controller(p_cam, quat_cam, q, dt=0.1):
    global _default
    if _default is None:
        _default = VisualServo()
    q_cmd, _ = _default.step(p_cam, q, quat_cam, dt)
    return q_cmd
