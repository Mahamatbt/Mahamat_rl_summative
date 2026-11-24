"""Shared utilities for all four algorithms: env factory, networks, evaluation, logging."""

from __future__ import annotations

import csv
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchrl.envs import Compose, GymWrapper, RewardScaling, RewardSum, StepCounter, TransformedEnv
from torchrl.modules import MLP

from environment import WaterDispatchEnv

warnings.filterwarnings("ignore")
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

# All algorithms train on rewards scaled by this factor (identical for every
# algorithm, so the comparison stays fair). Reported returns are always unscaled.
REWARD_SCALE = 0.1
EVAL_SEED_OFFSET = 100_000  # evaluation episodes never overlap training seeds


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed: int = 0, **env_kwargs) -> TransformedEnv:
    """TorchRL version of the custom Gymnasium env, used identically by every algorithm."""
    env = TransformedEnv(
        GymWrapper(WaterDispatchEnv(**env_kwargs)),
        Compose(StepCounter(), RewardSum(), RewardScaling(loc=0.0, scale=REWARD_SCALE)),
    )
    env.set_seed(seed)
    return env


def build_mlp(obs_dim: int, out_dim: int, hidden: list[int], activation: str = "relu") -> nn.Module:
    act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    return MLP(in_features=obs_dim, out_features=out_dim, num_cells=list(hidden), activation_class=act)


def env_dims() -> tuple[int, int]:
    env = WaterDispatchEnv()
    return env.observation_space.shape[0], env.action_space.n


@torch.no_grad()
def evaluate(net: nn.Module, n_episodes: int = 20, seed_offset: int = EVAL_SEED_OFFSET, **env_kwargs) -> dict:
    """Greedy evaluation on the raw Gymnasium env (argmax of Q-values / logits)."""
    env = WaterDispatchEnv(**env_kwargs)
    net.eval()
    returns, served, stranded, repairs = [], [], [], []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + i)
        done, ep_ret = False, 0.0
        while not done:
            a = int(net(torch.as_tensor(obs).unsqueeze(0)).argmax(-1).item())
            obs, r, term, trunc, info = env.step(a)
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
        served.append(100.0 * info["people_days_served"] / (env.pop.sum() * env.cfg.horizon_days))
        stranded.append(float(info["stranded"] or env.budget <= 0))
        repairs.append(info["repairs_done"])
    net.train()
    return {
        "eval_return": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_served_pct": float(np.mean(served)),
        "eval_failure_rate": float(np.mean(stranded)),
        "eval_repairs": float(np.mean(repairs)),
    }


def episode_returns(batch) -> list[float]:
    """Unscaled returns of the episodes that finished inside a collected batch."""
    done = batch["next", "done"].squeeze(-1)
    if not done.any():
        return []
    return batch["next", "episode_reward"][done].squeeze(-1).tolist()


class RunLogger:
    """Per-run learning curve (CSV) + console output."""

    def __init__(self, algo: str, run_name: str, fields: list[str]):
        self.dir = LOGS_DIR / algo
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_name}.csv"
        self.fields = fields
        self.f = open(self.path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()
        self.t0 = time.time()

    def log(self, row: dict, verbose: bool = True):
        self.w.writerow({k: row.get(k, "") for k in self.fields})
        self.f.flush()
        if verbose:
            msg = "  ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items() if k in self.fields
            )
            print(f"  [{time.time() - self.t0:6.0f}s] {msg}", flush=True)

    def close(self):
        self.f.close()


def append_summary(algo: str, row: dict):
    path = LOGS_DIR / algo / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("run") != row["run"]]
    rows.append({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_checkpoint(path: Path, algo: str, net: nn.Module, config: dict, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    obs_dim, n_actions = env_dims()
    torch.save(
        {
            "algo": algo,
            "state_dict": net.state_dict(),
            "config": config,
            "metrics": metrics,
            "obs_dim": obs_dim,
            "n_actions": n_actions,
        },
        path,
    )


def update_best(algo_dir: Path, algo: str, run_name: str, metrics: dict, ckpt: Path):
    """Keep models/<dir>/best_<algo>.pt pointing at the best run of the sweep."""
    meta_path = algo_dir / f"best_{algo}.json"
    best = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if best is None or metrics["eval_return"] > best["eval_return"] or best["run"] == run_name:
        (algo_dir / f"best_{algo}.pt").write_bytes(ckpt.read_bytes())
        meta_path.write_text(json.dumps({"run": run_name, **metrics}, indent=2))
        print(f"  -> new best {algo} model: {run_name} (eval return {metrics['eval_return']:.1f})")


def load_policy_net(path: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    net = build_mlp(ckpt["obs_dim"], ckpt["n_actions"], cfg["hidden"], cfg.get("activation", "relu"))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt
