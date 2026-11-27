import argparse
import json
import math
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
REPO = os.path.dirname(SIM)
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(REPO, "src", "visual_servo", "visual_servo"))

import servo_core as sc
from deploy.obs_builder import ObservationBuilder, e_psi_from_quat
from deploy.policy_np import NumpyPolicy

CONTROL_DT = 0.1
TICK_DT = 1.0 / 30.0
TICKS_PER_CONTROL = 3
MARKER_HALF = 0.03
V_MAX_JOINT = 2.0
D_GOAL = sc.D_GOAL
START_Q = np.array([0.0, math.pi / 2])
TAN_H = 324.964 / 606.542
TAN_V = 242.520 / 606.322
SCENARIOS = ("static", "lateral", "depth", "step", "wander", "near", "far", "edge", "edge_right", "jitter", "orbit", "waypoints")


def rot_to_quat_xyzw(r):
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return np.array([x, y, z, w])


def marker_rotation(normal_yaw):
    n = np.array([math.cos(normal_yaw), math.sin(normal_yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return np.stack((np.cross(up, n), up, n), axis=1)


class MarkerPath:
    def __init__(self, scenario, cam_pos, cam_rot, rng):
        self.s = scenario
        fwd = cam_rot[:, 2].copy()
        fwd[2] = 0.0
        fwd /= np.linalg.norm(fwd)
        self.fwd = fwd
        self.lat = np.array([-fwd[1], fwd[0], 0.0])
        self.p0 = cam_pos + D_GOAL * fwd
        self.yaw0 = math.atan2(-fwd[1], -fwd[0])
        self.rng = rng
        if scenario == "orbit":
            self.r0 = math.hypot(self.p0[0], self.p0[1])
            self.b0 = math.atan2(self.p0[1], self.p0[0])
            self.dir = 1.0 if rng.random() < 0.5 else -1.0
        if scenario == "waypoints":
            self.mount = sc.mount_transform()
            self.base, self.yaw = self.p0.copy(), self.yaw0
            self.hold = rng.uniform(0.0, 2.0)
            self._new_waypoint()
            self.jit_amp = rng.uniform(0.0, 0.02) if rng.random() < 0.5 else 0.0
            self.jit_f = rng.uniform(0.4, 1.5)
            self.jit_ph = rng.uniform(0, 2 * math.pi)
            jd = rng.normal(size=3)
            jd[2] *= 0.3
            self.jit_dir = jd / np.linalg.norm(jd)
            self.t_last = 0.0
        if scenario == "wander":
            self.amp = rng.uniform([0.03, 0.03, 0.0], [0.10, 0.08, math.radians(20)])
            self.freq = rng.uniform(0.03, 0.2, 3)
            self.phase = rng.uniform(0, 2 * math.pi, 3)

    def _new_waypoint(self):
        q = self.rng.uniform(sc.Q_MIN * 0.95, sc.Q_MAX * 0.95)
        tf = sc.camera_transform(q, self.mount)
        cp, cz, cx = tf[:3, 3], tf[:3, 2], tf[:3, 0]
        dist = self.rng.uniform(0.15, 0.65)
        lat = self.rng.uniform(-0.6, 0.6) * TAN_H * dist
        wp = cp + dist * cz + lat * cx
        wp[2] = cp[2] + self.rng.uniform(-0.04, 0.04)
        self.wp = wp
        self.wp_yaw = math.atan2(cz[1], cz[0]) + math.pi + self.rng.uniform(-math.radians(35), math.radians(35))
        self.speed = self.rng.uniform(0.02, 0.15)

    def _advance(self, t):
        dt = t - self.t_last
        self.t_last = t
        if self.hold > 0:
            self.hold = max(0.0, self.hold - dt)
            return
        delta = self.wp - self.base
        dist = np.linalg.norm(delta)
        step = min(self.speed * dt, dist)
        frac = step / max(dist, 1e-6)
        self.base = self.base + frac * delta
        dyaw = math.atan2(math.sin(self.wp_yaw - self.yaw), math.cos(self.wp_yaw - self.yaw))
        self.yaw += frac * dyaw
        if dist - step < 1e-4:
            self._new_waypoint()
            if self.rng.random() < 0.5:
                self.hold = self.rng.uniform(0.0, 3.0)

    def at(self, t):
        p, yaw = self.p0.copy(), self.yaw0
        if self.s == "lateral":
            p += 0.06 * math.sin(2 * math.pi * 0.15 * t) * self.lat
        elif self.s == "depth":
            p += 0.06 * math.sin(2 * math.pi * 0.10 * t) * self.fwd
        elif self.s == "step":
            p += (0.06 if 5.0 <= t < 12.0 else 0.0) * self.lat
        elif self.s in ("near", "far"):
            target = 0.12 if self.s == "near" else 0.50
            ramp = min(t / 5.0, 1.0) if t < 12.0 else max(1.0 - (t - 12.0) / 5.0, 0.0)
            p += (target - D_GOAL) * ramp * self.fwd
        elif self.s in ("edge", "edge_right"):
            p += (1.0 if self.s == "edge" else -1.0) * min(0.02 * t, 0.40) * self.lat
        elif self.s == "jitter":
            p += (0.015 * math.sin(2 * math.pi * 0.7 * t) + 0.005 * math.sin(2 * math.pi * 1.3 * t + 1.0)) * self.fwd
            p += 0.005 * math.sin(2 * math.pi * 0.9 * t + 2.0) * self.lat
        elif self.s == "orbit":
            b = self.b0 + self.dir * 0.15 * t
            p = np.array([self.r0 * math.cos(b), self.r0 * math.sin(b), self.p0[2]])
            yaw = b + math.pi
        elif self.s == "waypoints":
            self._advance(t)
            p = self.base + self.jit_amp * math.sin(2 * math.pi * self.jit_f * t + self.jit_ph) * self.jit_dir
            r = math.hypot(p[0], p[1])
            if r < 0.15:
                p[:2] *= 0.15 / max(r, 1e-6)
            yaw = self.yaw
        elif self.s == "wander":
            o = self.amp * np.sin(2 * math.pi * self.freq * t + self.phase) - self.amp * np.sin(self.phase)
            p += o[0] * self.lat + o[1] * self.fwd
            yaw += o[2]
        return p, marker_rotation(yaw)


class Actuator:
    def __init__(self, rng, ideal, stiction):
        self.ideal = ideal
        self.delay = int(rng.integers(1, 3))
        self.tau = np.array([rng.uniform(0.15, 0.25), rng.uniform(0.20, 0.50)])
        self.deadband = np.array([rng.uniform(0.003, 0.005), rng.uniform(0.0, 0.005)])
        self.stick_prob = rng.uniform(0.0, 0.05) if stiction else 0.0
        self.rng = rng

    def reset(self, q):
        self.queue = [np.array(q, dtype=float) for _ in range(self.delay + 1)]
        self.ref = np.array(q, dtype=float)
        self.applied = np.array(q, dtype=float)
        self.stick_left = 0

    def command(self, q_cmd):
        self.queue = [np.array(q_cmd, dtype=float)] + self.queue[:-1]
        self.applied = self.queue[self.delay]
        moving2 = abs(self.applied[1] - self.ref[1]) > self.deadband[1]
        if self.stick_left == 0 and moving2 and self.rng.random() < self.stick_prob:
            self.stick_left = int(self.rng.integers(2, 9))
        elif self.stick_left > 0:
            self.stick_left -= 1

    def physics(self, dt):
        if self.ideal:
            return self.applied
        err = self.applied - self.ref
        move = np.abs(err) > self.deadband
        if self.stick_left > 0:
            move[1] = False
        step = np.clip(np.minimum(dt / self.tau, 1.0) * err, -V_MAX_JOINT * dt, V_MAX_JOINT * dt)
        self.ref = self.ref + np.where(move, step, 0.0)
        return self.ref


class Sensor:
    def __init__(self, rng, noise, latency):
        self.rng = rng
        self.noise = noise
        self.latency = latency
        self.z_bias = rng.uniform(0.995, 1.005) if noise else 1.0
        self.psi_gain = rng.uniform(0.6, 1.0) if noise else 1.0
        self.psi_sigma = rng.uniform(math.radians(1), math.radians(5)) if noise else 0.0
        self.dropout = rng.uniform(0.0, 0.1) if noise else 0.0
        self.buf = []

    def measure(self, p_true, r_true, visible):
        if not visible or self.rng.random() < self.dropout:
            m = None
        else:
            p = p_true.copy()
            if self.noise:
                p += self.rng.normal(0, 0.002, 3)
                p[2] = p_true[2] * self.z_bias + self.rng.normal(0, 0.005 * p_true[2])
            n = r_true[:, 2]
            psi_true = math.atan2(n[0], -n[2])
            psi = psi_true * self.psi_gain + self.rng.normal(0, self.psi_sigma) if self.noise else psi_true
            if self.noise and abs(psi_true) < math.radians(8) and self.rng.random() < 0.5:
                psi = -psi
            m = (p, psi, rot_to_quat_xyzw(r_true))
        self.buf.append(m)
        if len(self.buf) > self.latency + 1:
            self.buf.pop(0)
        return self.buf[0]


def visible(p, r):
    z = p[2]
    if not (0.08 < z < 1.5):
        return False
    if abs(p[0]) + MARKER_HALF >= z * TAN_H or abs(p[1]) + MARKER_HALF >= z * TAN_V:
        return False
    return -r[2, 2] > math.cos(math.radians(70))


def make_controller(name, policy_path, q0, ik_kwargs=None):
    if name == "rl":
        pol = NumpyPolicy(policy_path)
        ob = ObservationBuilder(q0, d_goal=pol.d_goal, q_min=pol.q_min, q_max=pol.q_max, max_dq=pol.max_dq)
        state = {"q_cmd": np.array(q0, dtype=float)}

        def step(meas, q_meas, tick):
            if tick % TICKS_PER_CONTROL == 0:
                p, psi = (meas[0], meas[1]) if meas is not None else (None, 0.0)
                state["q_cmd"] = ob.apply(pol.act(ob.observe(p, psi, q_meas)))
            return state["q_cmd"].copy()

        return step
    if name == "ik":
        vs = sc.VisualServo(**(ik_kwargs or {}))
        vs.reset(q0)

        def step(meas, q_meas, tick):
            if meas is None:
                return vs.step(None, q_meas, dt=TICK_DT)[0]
            return vs.step(meas[0], q_meas, meas[2], dt=TICK_DT)[0]

        return step
    hold = np.array(q0, dtype=float)
    return lambda meas, q_meas, tick: hold.copy()


def run_episode(model_path, controller, scenario, seed, args, viewer=None, md=None):
    m, d = md if md is not None else (mujoco.MjModel.from_xml_path(model_path), None)
    d = d if d is not None else mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    d.qpos[:] = START_Q
    d.ctrl[:] = START_Q
    mujoco.mj_forward(m, d)
    site = m.site("color_optical").id
    mk = m.body("marker").mocapid[0]
    path = MarkerPath(scenario, d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3).copy(), rng)
    act = Actuator(rng, args.ideal_actuator, not args.no_stiction)
    act.reset(START_Q)
    sens = Sensor(rng, not args.no_noise, 0 if args.no_noise else TICKS_PER_CONTROL)
    ik_kwargs = json.loads(args.ik_args) if args.ik_args else {}
    ctrl = make_controller(controller, args.policy, START_Q, ik_kwargs)
    n_sub = int(round(TICK_DT / m.opt.timestep))
    ticks = int(round(args.duration / TICK_DT))
    log = {k: [] for k in ("t", "vis", "e_theta", "e_d", "e_psi", "q1", "q2", "q1_cmd", "q2_cmd", "marker_dist")}
    q_cmd = START_Q.copy()
    q_meas = START_Q.copy()
    for k in range(ticks):
        t0 = time.time()
        t = k * TICK_DT
        mp, mr = path.at(t)
        d.mocap_pos[mk] = mp
        w, x, y, z = rot_to_quat_xyzw(mr)[[3, 0, 1, 2]]
        d.mocap_quat[mk] = (w, x, y, z)
        mujoco.mj_forward(m, d)
        cp, cr = d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3).copy()
        p_true = cr.T @ (mp - cp)
        r_true = cr.T @ mr
        vis = visible(p_true, r_true)
        meas = sens.measure(p_true, r_true, vis)
        control_tick = k % TICKS_PER_CONTROL == 0
        if control_tick:
            q_meas = d.qpos[:2].copy()
            if not args.no_noise:
                q_meas = np.round((q_meas + rng.normal(0, 0.001, 2)) / (2 * math.pi / 16384)) * (2 * math.pi / 16384)
        q_cmd = ctrl(meas, q_meas, k)
        if control_tick:
            act.command(q_cmd)
        for _ in range(n_sub):
            d.ctrl[:] = act.physics(m.opt.timestep)
            mujoco.mj_step(m, d)
        if control_tick:
            n = r_true[:, 2]
            log["t"].append(t)
            log["vis"].append(float(vis))
            log["e_theta"].append(math.degrees(math.atan2(p_true[0], p_true[2])))
            log["e_d"].append(1000 * (p_true[2] - D_GOAL))
            log["e_psi"].append(math.degrees(math.atan2(n[0], -n[2])))
            log["q1"].append(float(d.qpos[0]))
            log["q2"].append(float(d.qpos[1]))
            log["marker_dist"].append(float(np.linalg.norm(p_true)))
            log["q1_cmd"].append(float(act.applied[0]))
            log["q2_cmd"].append(float(act.applied[1]))
        if viewer is not None:
            viewer.sync()
            if not viewer.is_running():
                break
            time.sleep(max(0.0, TICK_DT - (time.time() - t0)))
    return {k: np.array(v) for k, v in log.items()}


def summarize(log):
    vis = log["vis"] > 0.5
    late = log["t"] >= log["t"][-1] / 2
    dq = np.abs(np.diff(np.stack((log["q1_cmd"], log["q2_cmd"])), axis=1)).sum(0)

    def mean_abs(key, mask):
        return float(np.abs(log[key][mask]).mean()) if mask.any() else float("nan")

    def rms(key, mask):
        return float(np.sqrt(np.square(log[key][mask]).mean())) if mask.any() else float("nan")

    return {
        "visible_frac": float(vis.mean()),
        "abs_e_theta_deg": mean_abs("e_theta", vis),
        "rms_e_theta_deg": rms("e_theta", vis),
        "abs_e_d_mm": mean_abs("e_d", vis),
        "rms_e_d_mm": rms("e_d", vis),
        "abs_e_psi_deg": mean_abs("e_psi", vis),
        "abs_e_theta_deg_2nd_half": mean_abs("e_theta", vis & late),
        "abs_e_d_mm_2nd_half": mean_abs("e_d", vis & late),
        "mean_abs_dq_cmd": float(dq.mean()),
        "max_abs_dq_cmd": float(dq.max()),
        "q_travel_rad": float(np.abs(np.diff(np.stack((log["q1"], log["q2"])), axis=1)).sum()),
        "limit_frac": float(((np.abs(log["q1"]) > 0.98 * sc.Q_MAX[0]) | (np.abs(log["q2"]) > 0.98 * sc.Q_MAX[1])).mean()),
        "mean_abs_ddq_cmd": float(np.abs(np.diff(dq)).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=str, default=None)
    p.add_argument("--model", type=str, default=os.path.join(HERE, "twolink.xml"))
    p.add_argument("--controllers", type=str, default="rl,ik,zero")
    p.add_argument("--scenarios", type=str, default=",".join(SCENARIOS))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--ideal_actuator", action="store_true")
    p.add_argument("--no_noise", action="store_true")
    p.add_argument("--no_stiction", action="store_true")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--ik_args", type=str, default=None)
    args = p.parse_args()
    controllers = args.controllers.split(",")
    if "rl" in controllers and not args.policy:
        p.error("--policy is required for the rl controller")
    scenarios = args.scenarios.split(",")

    if args.viewer:
        run_viewer(args, controllers, scenarios)
        return

    res = {"args": vars(args), "results": {}}
    traces = {}
    for s in scenarios:
        for c in controllers:
            per_seed = [summarize(log) for log in (run_episode(args.model, c, s, seed, args) for seed in range(args.seeds))]
            res["results"][f"{s}/{c}"] = {k: float(np.nanmean([x[k] for x in per_seed])) for k in per_seed[0]}
            traces[(s, c)] = run_episode(args.model, c, s, 0, args)
    out_dir = args.out or os.path.join(SIM, "outputs", "sim2sim", time.strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for s in scenarios:
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for c in controllers:
            log = traces[(s, c)]
            for ax, key, label in zip(axes, ("e_theta", "e_d", "e_psi"), ("e_θ (deg)", "e_d (mm)", "e_ψ (deg)")):
                ax.plot(log["t"], np.where(log["vis"] > 0.5, log[key], np.nan), label=c)
                ax.set_ylabel(label)
                ax.axhline(0.0, color="grey", lw=0.5)
        axes[0].set_title(f"MuJoCo sim2sim, scenario {s}, seed 0")
        axes[0].legend()
        axes[-1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{s}.png"), dpi=110)
        plt.close(fig)

    width = max(len(k) for k in res["results"])
    print("SIM2SIM_RESULT", out_dir)
    print(f"{'scenario/controller':{width}s}  vis   |eθ|°  |ed|mm  |eψ|°  |dq|    travel  limit")
    for k, v in res["results"].items():
        print(f"{k:{width}s}  {v['visible_frac']:.2f}  {v['abs_e_theta_deg']:5.2f}  {v['abs_e_d_mm']:6.1f}  {v['abs_e_psi_deg']:5.1f}  {v['mean_abs_dq_cmd']:.4f}  {v['q_travel_rad']:6.2f}  {v['limit_frac']:.2f}")


def run_viewer(args, controllers, scenarios):
    import mujoco.viewer

    for c in controllers:
        for s in scenarios:
            m = mujoco.MjModel.from_xml_path(args.model)
            d = mujoco.MjData(m)
            print(f"[viewer] controller={c} scenario={s} (close the window to go to the next one)")
            with mujoco.viewer.launch_passive(m, d) as v:
                v.cam.lookat[:] = (0.15, 0.1, 0.15)
                v.cam.distance = 0.9
                v.cam.azimuth = -120
                v.cam.elevation = -25
                log = run_episode(args.model, c, s, 0, args, viewer=v, md=(m, d))
                print(json.dumps(summarize(log)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted")
        sys.stdout.flush()
        os._exit(130)
    sys.stdout.flush()
    os._exit(0)
