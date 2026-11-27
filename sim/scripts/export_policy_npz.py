import argparse
import glob
import os

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

run_dir = os.path.abspath(args.run)
ckpt = args.checkpoint or max(glob.glob(os.path.join(run_dir, "model_*.pt")), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
data = {"obs_mean": sd["actor_obs_normalizer._mean"].numpy()[0], "obs_std": sd["actor_obs_normalizer._std"].numpy()[0], "eps": np.array(1e-2)}
for i, l in enumerate(layers):
    data[f"w{i}"] = sd[f"actor.{l}.weight"].numpy()
    data[f"b{i}"] = sd[f"actor.{l}.bias"].numpy()
data["n_layers"] = np.array(len(layers))
import yaml

with open(os.path.join(run_dir, "params", "env.yaml")) as f:
    env_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
data["d_goal"] = np.array(env_cfg["d_goal"])
data["q_min"] = np.array(env_cfg["q_lower"])
data["q_max"] = np.array(env_cfg["q_upper"])
data["max_dq"] = np.array(env_cfg["max_dq"])
data["checkpoint"] = np.array(ckpt)
out = args.out or os.path.join(run_dir, "exported", "policy.npz")
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, **data)

actor = torch.nn.Sequential(*[m for i, l in enumerate(layers) for m in ([torch.nn.Linear(*data[f"w{i}"].shape[::-1])] + ([torch.nn.ELU()] if i < len(layers) - 1 else []))])
for i, l in enumerate(layers):
    actor[2 * i].weight.data = sd[f"actor.{l}.weight"]
    actor[2 * i].bias.data = sd[f"actor.{l}.bias"]
x = torch.randn(1000, data["w0"].shape[1]) * torch.tensor(data["obs_std"]) + torch.tensor(data["obs_mean"])
ref = actor((x - torch.tensor(data["obs_mean"])) / (torch.tensor(data["obs_std"]) + 1e-2)).detach().numpy()

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.policy_np import NumpyPolicy

got = NumpyPolicy(out).act_raw(x.numpy())
print("EXPORT", out, "from", ckpt, "max_abs_diff", float(np.abs(got - ref).max()))
