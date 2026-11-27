import numpy as np


class NumpyPolicy:
    def __init__(self, path):
        d = np.load(path)
        self.d_goal = float(d["d_goal"]) if "d_goal" in d else 0.30
        self.max_dq = float(d["max_dq"]) if "max_dq" in d else 0.1
        self.q_min = d["q_min"].astype(np.float64) if "q_min" in d else np.array([-1.57, -2.5])
        self.q_max = d["q_max"].astype(np.float64) if "q_max" in d else np.array([1.57, 2.5])
        self.mean = d["obs_mean"].astype(np.float64)
        self.std = d["obs_std"].astype(np.float64)
        self.eps = float(d["eps"])
        n = int(d["n_layers"])
        self.w = [d[f"w{i}"].astype(np.float64) for i in range(n)]
        self.b = [d[f"b{i}"].astype(np.float64) for i in range(n)]

    def act_raw(self, obs):
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / (self.std + self.eps)
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            x = x @ w.T + b
            if i < len(self.w) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
        return x

    def act(self, obs):
        return np.clip(self.act_raw(obs), -1.0, 1.0)
