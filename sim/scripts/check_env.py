import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--realtime", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SIM_DIR)

from twolink_rl import geometry as geo
from twolink_rl.marker_track_env import MarkerTrackEnv, MarkerTrackEnvCfg


MOUNT_NOMINAL = MarkerTrackEnvCfg().mount_pos_nominal
D_GOAL = MarkerTrackEnvCfg().d_goal
MAX_DQ = MarkerTrackEnvCfg().max_dq


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def p_controller(obs, gain=0.2, damping=0.02):
    seen = obs[:, 4] > 0.5
    p = obs[:, 0:3].clone()
    p[:, 2] += D_GOAL
    q = obs[:, 6:8]
    n = obs.shape[0]
    mount = torch.tensor(MOUNT_NOMINAL, device=obs.device).repeat(n, 1)
    yaw0 = torch.zeros(n, device=obs.device)
    cam_pos, cam_rot = geo.camera_pose(q, mount, yaw0)
    marker_w = cam_pos + (cam_rot @ p.unsqueeze(-1)).squeeze(-1)

    def errs(qq):
        cp, cr = geo.camera_pose(qq, mount, yaw0)
        pc = (cr.transpose(-1, -2) @ (marker_w - cp).unsqueeze(-1)).squeeze(-1)
        return torch.stack((torch.atan2(pc[:, 0], pc[:, 2]), pc[:, 2] - D_GOAL), dim=-1)

    e = errs(q)
    h = 1e-4
    cols = []
    for j in range(2):
        dq = torch.zeros_like(q)
        dq[:, j] = h
        cols.append((errs(q + dq) - e) / h)
    jac = torch.stack(cols, dim=-1)
    w = torch.diag(torch.tensor([1.0, 3.0], device=obs.device))
    jw = w @ jac
    a = jw.transpose(-1, -2) @ jw + damping * torch.eye(2, device=obs.device)
    dq = -gain * torch.linalg.solve(a, (jw.transpose(-1, -2) @ (w @ e.unsqueeze(-1)))).squeeze(-1)
    scale = (dq.abs().max(dim=-1, keepdim=True).values / MAX_DQ).clamp(min=1.0)
    dq = dq / scale
    dq = torch.where(seen.unsqueeze(-1), dq, torch.zeros_like(dq))
    return dq / MAX_DQ


def run_policy(env, policy, steps):
    obs, _ = env.reset()
    obs = obs["policy"]
    total, vis, et, ed, ep, nan = 0.0, 0.0, 0.0, 0.0, 0.0, False
    o_min = torch.full_like(obs[0], float("inf"))
    o_max = torch.full_like(obs[0], float("-inf"))
    for _ in range(steps):
        act = policy(obs)
        t_step = time.time()
        obs, rew, term, trunc, extras = env.step(act)
        if args.realtime:
            time.sleep(max(0.0, env.step_dt - (time.time() - t_step)))
        obs = obs["policy"]
        nan |= bool(torch.isnan(obs).any() or torch.isnan(rew).any())
        o_min = torch.minimum(o_min, obs.min(0).values)
        o_max = torch.maximum(o_max, obs.max(0).values)
        v = env.is_visible.float()
        n = v.sum().clamp(min=1)
        total += rew.mean().item()
        vis += v.mean().item()
        et += ((v * torch.rad2deg(env.e_theta.abs())).sum() / n).item()
        ed += ((v * 1000 * env.e_d.abs()).sum() / n).item()
        ep += ((v * torch.rad2deg(env.e_psi.abs())).sum() / n).item()
    return {
        "return_per_step": total / steps,
        "visible_frac": vis / steps,
        "abs_e_theta_deg": et / steps,
        "abs_e_d_mm": ed / steps,
        "abs_e_psi_deg": ep / steps,
        "nan": nan,
        "obs_min": [round(x, 4) for x in o_min.tolist()],
        "obs_max": [round(x, 4) for x in o_max.tolist()],
    }


def main():
    cfg = MarkerTrackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.show_marker = not args.headless
    cfg.viewer.eye = (0.9, -0.9, 0.7)
    cfg.viewer.lookat = (0.15, 0.0, 0.15)
    env = MarkerTrackEnv(cfg)
    dev = env.device
    res = {"num_envs": args.num_envs, "steps": args.steps}

    env.reset()
    link2 = env.robot.find_bodies("link2")[0][0]
    res["body_names"] = env.robot.body_names
    res["joint_names"] = env.robot.joint_names
    fk_err_pos, fk_err_yaw, track_err = [], [], []
    for _ in range(5):
        tgt = torch.rand(env.num_envs, 2, device=dev) * (env.q_upper - env.q_lower) * 0.9 + env.q_lower * 0.9
        for _ in range(60):
            env.q_cmd[:] = tgt
            env.cmd_queue[:] = tgt.unsqueeze(1)
            env.step(torch.zeros(env.num_envs, 2, device=dev))
        q = env.robot.data.joint_pos[:, env._joint_ids]
        pos_w = env.robot.data.body_pos_w[:, link2] - env.scene.env_origins
        quat = env.robot.data.body_quat_w[:, link2]
        yaw = torch.atan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]), 1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
        p_fk, _ = geo.link2_pose(q)
        fk_err_pos.append((pos_w - p_fk).norm(dim=-1).max().item())
        fk_err_yaw.append(wrap(yaw - (q[:, 0] + q[:, 1])).abs().max().item())
        track_err.append((q - tgt).abs().max().item())
    res["fk_max_pos_err_m"] = max(fk_err_pos)
    res["fk_max_yaw_err_rad"] = max(fk_err_yaw)
    res["hold_max_joint_err_rad"] = max(track_err)

    env.reset()
    for _ in range(20):
        env.step(torch.zeros(env.num_envs, 2, device=dev))
    q0 = env.robot.data.joint_pos[:, env._joint_ids].clone()
    cmd0 = env.q_cmd.clone()
    step_act = torch.zeros(env.num_envs, 2, device=dev)
    step_act[:, 0] = 1.0
    step_act[:, 1] = -1.0
    n_resp = 30
    ys = []
    for i in range(n_resp):
        env.step(step_act if i == 0 else torch.zeros(env.num_envs, 2, device=dev))
        ys.append((env.robot.data.joint_pos[:, env._joint_ids] - q0) / (env.q_cmd - cmd0))
    y = torch.stack(ys, dim=0)
    t = (torch.arange(n_resp, device=dev).float() + 1) * env.step_dt
    for name, th in (("t10_s", 0.1), ("t63_s", 0.63), ("t90_s", 0.9)):
        hit = y >= th
        first = torch.where(hit.any(0), t[hit.float().argmax(0)], torch.full_like(y[0], float("nan")))
        res["step_" + name] = [round(x, 3) for x in first.nanmedian(0).values.tolist()]
    res["step_final_frac"] = [round(x, 4) for x in y[-1].median(0).values.tolist()]
    res["step_mean_curve"] = [[round(x, 3) for x in r] for r in y.mean(1).tolist()]
    res["cmd_after_step"] = [round(x, 4) for x in (env.q_cmd - cmd0).mean(0).tolist()]

    zero = lambda o: torch.zeros(o.shape[0], 2, device=dev)
    rand = lambda o: 2 * torch.rand(o.shape[0], 2, device=dev) - 1
    t0 = time.time()
    res["zero_action"] = run_policy(env, zero, args.steps)
    res["random_action"] = run_policy(env, rand, args.steps)
    res["p_controller"] = run_policy(env, p_controller, args.steps)
    res["env_steps_per_s_p_controller"] = 3 * args.steps * args.num_envs / (time.time() - t0)
    res["gpu_mem_max_mb"] = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None

    out = args.out or os.path.join(SIM_DIR, "outputs", "check_env", time.strftime("%Y-%m-%d_%H%M%S") + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("CHECK_ENV_RESULT", out)
    print(json.dumps(res, indent=1))
    env.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
