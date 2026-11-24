"""Deep Q-Network training (TorchRL) with a 10-configuration hyperparameter sweep.

Usage:
    uv run python -m training.dqn_training                 # all 10 runs
    uv run python -m training.dqn_training --runs 0 3 7    # selected runs
    uv run python -m training.dqn_training --frames 50000  # shorter runs
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field

import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate, ValueEstimators

from training.common import (
    MODELS_DIR,
    RunLogger,
    append_summary,
    build_mlp,
    episode_returns,
    evaluate,
    make_env,
    save_checkpoint,
    set_seed,
    update_best,
)


@dataclass
class DQNConfig:
    name: str
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 50_000
    eps_end: float = 0.05
    eps_decay_frac: float = 0.4  # fraction of training over which epsilon anneals 1.0 -> eps_end
    target_update: str = "hard"  # "hard" or "soft"
    target_interval: int = 500  # gradient steps between hard target updates
    tau: float = 0.005  # soft update rate
    double_dqn: bool = False
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    frames_per_batch: int = 200
    updates_per_batch: int = 50
    warmup_frames: int = 5_000
    loss_function: str = "smooth_l1"
    max_grad_norm: float = 10.0


# 10 hyperparameter combinations: start from a baseline, change one or two things at a time
SWEEP = [
    DQNConfig("dqn_01_baseline"),
    DQNConfig("dqn_02_lr_1e-3", lr=1e-3),
    DQNConfig("dqn_03_lr_1e-4", lr=1e-4),
    DQNConfig("dqn_04_gamma_0.95", gamma=0.95),
    DQNConfig("dqn_05_gamma_0.999", gamma=0.999),
    DQNConfig("dqn_06_batch256_buf100k", batch_size=256, buffer_size=100_000),
    DQNConfig("dqn_07_batch64_buf10k", batch_size=64, buffer_size=10_000),
    DQNConfig("dqn_08_fast_explore", eps_decay_frac=0.15, eps_end=0.01),
    DQNConfig("dqn_09_double_soft", double_dqn=True, target_update="soft", tau=0.005),
    DQNConfig("dqn_10_double_big", double_dqn=True, hidden=[256, 256], target_interval=1000, lr=3e-4),
]


def train(cfg: DQNConfig, total_frames: int, seed: int = 0, eval_every: int = 10_000, verbose: bool = True) -> dict:
    set_seed(seed)
    env = make_env(seed)
    obs_dim = env.observation_spec["observation"].shape[-1]
    n_actions = env.action_spec.shape[-1]

    q_net = build_mlp(obs_dim, n_actions, cfg.hidden, cfg.activation)
    actor = QValueActor(q_net, in_keys=["observation"], spec=env.action_spec)
    explore = EGreedyModule(
        env.action_spec,
        eps_init=1.0,
        eps_end=cfg.eps_end,
        annealing_num_steps=int(cfg.eps_decay_frac * total_frames),
    )
    policy = TensorDictSequential(actor, explore)

    loss_module = DQNLoss(actor, loss_function=cfg.loss_function, delay_value=True, double_dqn=cfg.double_dqn)
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=cfg.gamma)
    if cfg.target_update == "hard":
        target_updater = HardUpdate(loss_module, value_network_update_interval=cfg.target_interval)
    else:
        target_updater = SoftUpdate(loss_module, tau=cfg.tau)
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)

    buffer = ReplayBuffer(storage=LazyTensorStorage(cfg.buffer_size), batch_size=cfg.batch_size)
    collector = Collector(
        env,
        policy,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=total_frames,
        init_random_frames=cfg.warmup_frames,
    )

    logger = RunLogger(
        "dqn",
        cfg.name,
        ["frames", "train_return", "loss", "q_mean", "epsilon", "eval_return", "eval_served_pct", "eval_failure_rate"],
    )
    frames, recent, next_eval = 0, [], eval_every
    best = {"eval_return": -float("inf")}
    ckpt_path = MODELS_DIR / "dqn" / f"{cfg.name}.pt"
    loss_val = q_mean = float("nan")
    t0 = time.time()

    for batch in collector:
        frames += batch.numel()
        recent = (recent + episode_returns(batch))[-20:]
        buffer.extend(batch.reshape(-1))
        if frames >= cfg.warmup_frames:
            explore.step(batch.numel())
            for _ in range(cfg.updates_per_batch):
                sample = buffer.sample()
                out = loss_module(sample)
                optim.zero_grad()
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
                optim.step()
                target_updater.step()
            loss_val = out["loss"].item()
            with torch.no_grad():
                q_mean = q_net(sample["observation"]).max(-1).values.mean().item() / 0.1  # unscaled
        collector.update_policy_weights_()

        if frames >= next_eval or frames >= total_frames:
            next_eval += eval_every
            metrics = evaluate(q_net)
            row = {
                "frames": frames,
                "train_return": sum(recent) / max(len(recent), 1),
                "loss": loss_val,
                "q_mean": q_mean,
                "epsilon": float(explore.eps.item()),
                **metrics,
            }
            logger.log(row, verbose)
            if metrics["eval_return"] > best["eval_return"]:
                best = {**metrics, "frames": frames}
                save_checkpoint(ckpt_path, "dqn", q_net, asdict(cfg), best)

    collector.shutdown()
    logger.close()
    summary = {"run": cfg.name, **asdict(cfg), **best, "minutes": (time.time() - t0) / 60, "seed": seed}
    append_summary("dqn", summary)
    update_best(MODELS_DIR / "dqn", "dqn", cfg.name, best, ckpt_path)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, nargs="*", help="indices (0-9) of sweep configs to run; default all")
    p.add_argument("--frames", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    runs = args.runs if args.runs else range(len(SWEEP))
    for i in runs:
        cfg = SWEEP[i]
        print(f"\n=== DQN run {i + 1}/{len(SWEEP)}: {cfg.name} ===")
        s = train(cfg, args.frames, args.seed, verbose=not args.quiet)
        print(f"=== {cfg.name}: best eval return {s['eval_return']:.1f}, served {s['eval_served_pct']:.1f}% ===")


if __name__ == "__main__":
    main()
