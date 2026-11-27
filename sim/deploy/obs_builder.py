import math

import numpy as np

D_GOAL = 0.20
Q_MIN = np.array([-3.0, -3.0])
Q_MAX = np.array([3.0, 3.0])
MAX_DQ = 0.2


def e_psi_from_quat(quat_xyzw):
    x, y, z, w = quat_xyzw
    n_x = 2 * (x * z + w * y)
    n_z = 1 - 2 * (x * x + y * y)
    return math.atan2(n_x, -n_z)


class ObservationBuilder:
    def __init__(self, q0, d_goal=D_GOAL, q_min=Q_MIN, q_max=Q_MAX, max_dq=MAX_DQ):
        self.d_goal = d_goal
        self.q_min = np.asarray(q_min, dtype=float)
        self.q_max = np.asarray(q_max, dtype=float)
        self.max_dq = max_dq
        self.reset(q0)

    def reset(self, q0):
        self.q_cmd = np.clip(np.asarray(q0, dtype=float), self.q_min, self.q_max)
        self.last_seen = np.zeros(4)
        self.steps_unseen = 0

    def observe(self, p_cam, e_psi, q_meas):
        seen = p_cam is not None
        if seen:
            self.last_seen = np.array([p_cam[0], p_cam[1], p_cam[2], e_psi], dtype=float)
            self.steps_unseen = 0
        else:
            self.steps_unseen += 1
        return np.concatenate((
            self.last_seen[0:2],
            [self.last_seen[2] - self.d_goal, self.last_seen[3], 1.0 if seen else 0.0, min(self.steps_unseen / 10.0, 1.0)],
            np.asarray(q_meas, dtype=float),
            self.q_cmd,
        ))

    def apply(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.q_cmd = np.clip(self.q_cmd + self.max_dq * a, self.q_min, self.q_max)
        return self.q_cmd.copy()
