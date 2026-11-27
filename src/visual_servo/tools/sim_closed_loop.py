#!/usr/bin/env python3
import argparse
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from visual_servo import servo_core as sc

DT = 0.005
CMD_DELAY = 0.12
STATE_DELAY = 0.05
IMAGE_DELAY = 0.05
TAU = (0.25, 0.35)
STICTION = (0.004, 0.005)
NOISE_XY = 0.001
NOISE_Z = 0.003


class Robot:
    def __init__(self, q, rng):
        self.q = np.array(q, dtype=float)
        self.target = self.q.copy()
        self.moving = np.array([False, False])
        self.rng = rng

    def update(self, dt):
        err = self.target - self.q
        for j in range(2):
            if not self.moving[j] and abs(err[j]) > STICTION[j]:
                self.moving[j] = True
            if self.moving[j]:
                self.q[j] += err[j] * (1 - math.exp(-dt / TAU[j]))
                if abs(self.target[j] - self.q[j]) < STICTION[j] * 0.8:
                    self.moving[j] = False


SCENARIOS = {'step': 12.0, 'sine': 30.0, 'fast': 20.0, 'near': 14.0, 'far': 14.0, 'edge': 16.0, 'jitter': 20.0}
MARKER_HALF = 0.03
FOV = (math.radians(43.5), math.radians(33.0))


def ramp(t, t0, t1):
    return min(1.0, max(0.0, (t - t0) / (t1 - t0)))


def marker_path(kind, m0, normal_yaw, t):
    m = m0.copy()
    left = np.array([-math.sin(normal_yaw), math.cos(normal_yaw), 0.0])
    away = np.array([math.cos(normal_yaw), math.sin(normal_yaw), 0.0])
    if kind == 'step':
        m += (0.04 if t >= 1.0 else 0.0) * left + (0.05 if t >= 6.0 else 0.0) * away
    elif kind == 'sine':
        m += 0.06 * math.sin(2 * math.pi * 0.15 * t) * left + 0.04 * math.sin(2 * math.pi * 0.07 * t) * away
    elif kind == 'fast':
        m += 0.08 * math.sin(2 * math.pi * 0.3 * t) * left + 0.03 * math.sin(2 * math.pi * 0.2 * t) * away
    elif kind == 'near':
        radial = m0 / np.linalg.norm(m0[:2]) * np.array([1.0, 1.0, 0.0])
        m -= 0.17 * (ramp(t, 1.0, 4.0) - ramp(t, 9.0, 12.0)) * radial
    elif kind == 'far':
        m += 0.40 * (ramp(t, 1.0, 4.0) - ramp(t, 9.0, 12.0)) * away
    elif kind == 'edge':
        m += 0.03 * max(0.0, t - 1.0) * left
    elif kind == 'jitter':
        m += (0.015 * math.sin(2 * math.pi * 0.7 * t) + 0.005 * math.sin(2 * math.pi * 1.3 * t + 1.0)) * away
        m += 0.005 * math.sin(2 * math.pi * 0.9 * t + 2.0) * left
    return m


def visible(p):
    if p[2] <= 0.07:
        return False
    margin = math.atan2(MARKER_HALF, p[2])
    return abs(math.atan2(p[0], p[2])) < FOV[0] / 2 - margin and abs(math.atan2(p[1], p[2])) < FOV[1] / 2 - margin


def simulate(kind, gain, duration, seed=0, deadband=sc.DEADBAND, make_servo=None):
    rng = np.random.default_rng(seed)
    q0 = np.array([0.0, math.pi / 2])
    servo = make_servo(gain) if make_servo else sc.VisualServo()
    cam0 = sc.camera_transform(q0, servo.mount)
    m0 = (cam0 @ np.array([0.0, 0.0, servo.d_goal, 1.0]))[:3]
    yaw0 = math.atan2(cam0[1, 2], cam0[0, 2])
    robot = Robot(q0, rng)
    servo.reset(q0)
    rate = getattr(servo, 'CONTROL_RATE', 10.0)
    kw = {'dt': 1.0 / rate} if rate != 10.0 else {}
    used_img = None
    hist_q, hist_t = [], []
    pending, images = [], []
    log = []
    next_ctrl, next_img, next_tick = 0.0, 0.0, rng.uniform(0, 0.1)
    t = 0.0
    while t < duration:
        m = marker_path(kind, m0, yaw0, t)
        hist_t.append(t)
        hist_q.append(robot.q.copy())
        if t >= next_img:
            p = np.linalg.inv(sc.camera_transform(robot.q, servo.mount)) @ np.append(m, 1.0)
            if visible(p[:3]):
                p = p[:3] + rng.normal(0, [NOISE_XY, NOISE_XY, NOISE_Z * p[2] / 0.3])
                images.append((t + IMAGE_DELAY, t, p))
            next_img += 1 / 30
        if t >= next_tick:
            ready = [c for c in pending if c[0] <= t]
            if ready:
                robot.target = ready[-1][1].copy()
                pending = [c for c in pending if c[0] > t]
            next_tick += 0.1
        if t >= next_ctrl:
            seen = [im for im in images if im[0] <= t]
            state_idx = max(0, int((t - STATE_DELAY) / DT) - 1)
            q_state = hist_q[min(state_idx, len(hist_q) - 1)]
            fresh = seen and t - seen[-1][0] <= 0.3 and (rate <= 10.0 or seen[-1][1] != used_img)
            if fresh:
                _, t_img, p = seen[-1]
                used_img = t_img
                q_img = hist_q[min(int(t_img / DT), len(hist_q) - 1)]
                q_cmd, info = servo.step(p, q_img, **kw)
            else:
                q_cmd, info = servo.step(None, q_state, **kw)
            pending.append((t + CMD_DELAY - 0.035, q_cmd.copy()))
            true_p = (np.linalg.inv(sc.camera_transform(robot.q, servo.mount)) @ np.append(m, 1.0))[:3]
            e_theta, e_d, _ = sc.marker_errors(true_p, d_goal=servo.d_goal)
            log.append((t, e_theta, e_d, q_cmd[0], q_cmd[1], robot.q[0], robot.q[1], float(visible(true_p))))
            next_ctrl += 1.0 / rate
        robot.update(DT)
        t += DT
    return np.array(log)


def metrics(log):
    th, d = np.degrees(log[:, 1]), log[:, 2] * 1e3
    travel = np.abs(np.diff(log[:, 5:7], axis=0)).sum()
    step = np.diff(log[:, 0]).mean()
    acc = np.diff(log[:, 5:7], n=2, axis=0) / step ** 2
    at_limit = np.any(np.abs(log[:, 5:7]) > 0.98 * sc.Q_MAX, axis=1).mean()
    return dict(rms_theta=np.sqrt(np.mean(th ** 2)), max_theta=np.abs(th).max(), rms_d=np.sqrt(np.mean(d ** 2)),
                travel=travel, limit=at_limit, visible=log[:, 7].mean(),
                rough=np.sqrt(np.mean(acc ** 2)))


def load_controller(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('servo_alt', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', default=','.join(SCENARIOS))
    parser.add_argument('--compare', help='path to another servo_core.py to run side by side (e.g. v1)')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--out', default='sim_closed_loop')
    args = parser.parse_args()
    ctrls = {'current': lambda g: sc.VisualServo()}
    if args.compare:
        alt = load_controller(args.compare)
        ctrls['compare'] = lambda g: alt.VisualServo(d_goal=sc.D_GOAL)
    kinds = args.scenarios.split(',')
    rows = []
    fig, axes = plt.subplots(3, len(kinds), figsize=(4.2 * len(kinds), 8), sharex='col', squeeze=False)
    for col, kind in enumerate(kinds):
        for name, make in ctrls.items():
            ms = []
            for seed in range(args.seeds):
                log = simulate(kind, None, SCENARIOS[kind], seed, make_servo=make)
                ms.append(metrics(log))
                if seed == 0:
                    axes[0][col].plot(log[:, 0], np.degrees(log[:, 1]), lw=1, label=name)
                    axes[1][col].plot(log[:, 0], log[:, 2] * 1e3, lw=1, label=name)
                    axes[2][col].plot(log[:, 0], log[:, 5], lw=1, label=f'{name} q1')
                    axes[2][col].plot(log[:, 0], log[:, 6], lw=1, ls='--', label=f'{name} q2')
            rows.append((kind, name, {k: np.mean([m[k] for m in ms]) for k in ms[0]}))
        axes[0][col].set_title(kind)
        for ax, label in zip(axes[:, col], ('e_theta (deg)', 'e_d (mm)', 'q (rad)')):
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(label)
        for lim in (sc.Q_MIN[0], sc.Q_MAX[0], sc.Q_MIN[1], sc.Q_MAX[1]):
            axes[2][col].axhline(lim, color='0.7', lw=0.6, ls=':')
        axes[0][col].legend(fontsize=7)
        axes[2][col].legend(fontsize=6)
        axes[2][col].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(args.out + '.png', dpi=120)
    lines = [f'd_goal {sc.D_GOAL} m, {args.seeds} seeds; travel = sum |dq| of the robot; '
             'limit = fraction of time with a joint above 98 % of its limit; visible = marker in the FOV', '',
             '| scenario | controller | RMS e_theta (deg) | max e_theta (deg) | RMS e_d (mm) | joint travel (rad) | RMS joint accel (rad/s2) | at limit (%) | visible (%) |',
             '|---|---|---|---|---|---|---|---|---|']
    lines += [f"| {k} | {n} | {m['rms_theta']:.2f} | {m['max_theta']:.1f} | {m['rms_d']:.1f} | {m['travel']:.2f} | {m['rough']:.1f} | "
              f"{100 * m['limit']:.0f} | {100 * m['visible']:.0f} |" for k, n, m in rows]
    with open(args.out + '.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'Saved {args.out}.png, {args.out}.md')


if __name__ == '__main__':
    main()
