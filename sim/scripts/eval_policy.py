import argparse
import glob
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=190)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--mount", type=float, nargs=3, default=None)
parser.add_argument("--tag", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SIM_DIR)
sys.path.insert(0, os.path.join(SIM_DIR, "scripts"))

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit
from rsl_rl.runners import OnPolicyRunner

from twolink_rl.agents.rsl_rl_ppo_cfg import MarkerTrackPPORunnerCfg
from twolink_rl.marker_track_env import MarkerTrackEnv, MarkerTrackEnvCfg


MOUNT_NOMINAL = MarkerTrackEnvCfg().mount_pos_nominal
D_GOAL = MarkerTrackEnvCfg().d_goal
MAX_DQ = MarkerTrackEnvCfg().max_dq


def jacobian_controller(obs, gain=0.2, damping=0.02):
    from twolink_rl import geometry as geo

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
    cols = []
    for j in range(2):
        dq = torch.zeros_like(q)
        dq[:, j] = 1e-4
        cols.append((errs(q + dq) - e) / 1e-4)
    jac = torch.stack(cols, dim=-1)
    w = torch.diag(torch.tensor([1.0, 3.0], device=obs.device))
    jw = w @ jac
    a = jw.transpose(-1, -2) @ jw + damping * torch.eye(2, device=obs.device)
    dq = -gain * torch.linalg.solve(a, jw.transpose(-1, -2) @ (w @ e.unsqueeze(-1))).squeeze(-1)
    dq = dq / (dq.abs().max(dim=-1, keepdim=True).values / MAX_DQ).clamp(min=1.0)
    return torch.where(seen.unsqueeze(-1), dq, torch.zeros_like(dq)) / MAX_DQ


def rollout(env, policy, steps, seed):
    with torch.inference_mode():
        return _rollout(env, policy, steps, seed)


def _rollout(env, policy, steps, seed):
    torch.manual_seed(seed)
    obs, _ = env.reset()
    obs = obs["policy"]
    keys = ("vis", "e_theta", "e_d", "e_psi", "dq_cmd", "rew")
    hist = {k: [] for k in keys}
    prev_cmd = env.q_cmd.clone()
    with torch.inference_mode():
        for _ in range(steps):
            obs, rew, _, _, _ = env.step(policy(obs))
            obs = obs["policy"]
            hist["vis"].append(env.is_visible.float())
            hist["e_theta"].append(torch.rad2deg(env.e_theta))
            hist["e_d"].append(1000 * env.e_d)
            hist["e_psi"].append(torch.rad2deg(env.e_psi))
            hist["dq_cmd"].append((env.q_cmd - prev_cmd).abs().sum(-1))
            hist["rew"].append(rew)
            prev_cmd = env.q_cmd.clone()
    h = {k: torch.stack(v, 0) for k, v in hist.items()}
    vis = h["vis"]
    n = vis.sum().clamp(min=1)
    late = slice(steps // 2, None)
    vl = vis[late]
    nl = vl.sum().clamp(min=1)
    dd = h["dq_cmd"][1:] - h["dq_cmd"][:-1]
    summary = {
        "return_per_step": h["rew"].mean().item(),
        "visible_frac": vis.mean().item(),
        "visible_frac_2nd_half": vl.mean().item(),
        "abs_e_theta_deg": ((vis * h["e_theta"].abs()).sum() / n).item(),
        "abs_e_d_mm": ((vis * h["e_d"].abs()).sum() / n).item(),
        "abs_e_psi_deg": ((vis * h["e_psi"].abs()).sum() / n).item(),
        "abs_e_theta_deg_2nd_half": ((vl * h["e_theta"][late].abs()).sum() / nl).item(),
        "abs_e_d_mm_2nd_half": ((vl * h["e_d"][late].abs()).sum() / nl).item(),
        "abs_e_psi_deg_2nd_half": ((vl * h["e_psi"][late].abs()).sum() / nl).item(),
        "mean_abs_dq_cmd_rad_per_step": h["dq_cmd"].mean().item(),
        "mean_abs_ddq_cmd_rad_per_step": dd.abs().mean().item(),
    }
    return summary, {k: v[:, :4].cpu() for k, v in h.items()}


def main():
    run_dir = os.path.abspath(args.run)
    ckpt = args.checkpoint or max(glob.glob(os.path.join(run_dir, "model_*.pt")), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    cfg = MarkerTrackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    if args.mount is not None:
        cfg.mount_pos_nominal = tuple(args.mount)
    global MOUNT_NOMINAL
    MOUNT_NOMINAL = cfg.mount_pos_nominal
    env = MarkerTrackEnv(cfg)
    agent_cfg = MarkerTrackPPORunnerCfg()
    runner = OnPolicyRunner(RslRlVecEnvWrapper(env), agent_cfg.to_dict(), log_dir=None, device=env.device)
    runner.load(ckpt)
    nn = runner.alg.policy
    export_dir = os.path.join(run_dir, "exported")
    export_policy_as_jit(nn, normalizer=getattr(nn, "actor_obs_normalizer", None), path=export_dir, filename="policy.pt")
    jit = torch.jit.load(os.path.join(export_dir, "policy.pt"), map_location=env.device)
    jit.eval()

    dev = env.device
    controllers = {
        "rl_policy": lambda o: jit(o),
        "jacobian_test_controller": jacobian_controller,
        "zero_action": lambda o: torch.zeros(o.shape[0], 2, device=dev),
    }
    res = {"run": run_dir, "checkpoint": ckpt, "mount_pos_nominal": list(cfg.mount_pos_nominal), "mount_pos_range": list(cfg.mount_pos_range), "num_envs": args.num_envs, "steps": args.steps, "seed": args.seed}
    traces = {}
    for name, fn in controllers.items():
        res[name], traces[name] = rollout(env, fn, args.steps, args.seed)

    out_dir = os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
    out = os.path.join(out_dir, f"eval_{stamp}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    torch.save(traces, os.path.join(out_dir, f"traces_{stamp}.pt"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [0.1 * (i + 1) for i in range(args.steps)]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for name, style in (("rl_policy", "-"), ("jacobian_test_controller", "--")):
        tr = traces[name]
        for ax, key, label in zip(axes, ("e_theta", "e_d", "e_psi"), ("e_θ (deg)", "e_d (mm)", "e_ψ (deg)")):
            y = torch.where(tr["vis"][:, 0] > 0.5, tr[key][:, 0], torch.tensor(float("nan")))
            ax.plot(t, y.tolist(), style, label=name)
            ax.set_ylabel(label)
            ax.axhline(0.0, color="grey", lw=0.5)
    axes[0].legend()
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"trace_env0_{stamp}.png"), dpi=120)
    print("EVAL_RESULT", out)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
