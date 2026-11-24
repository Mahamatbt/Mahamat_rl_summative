"""Policy-gradient training (TorchRL): REINFORCE, PPO and A2C, 10-run sweeps each.

Usage:
    uv run python -m training.pg_training --algo ppo              # 10 PPO runs
    uv run python -m training.pg_training --algo all              # 30 runs
    uv run python -m training.pg_training --algo a2c --runs 0 4   # selected runs
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE

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

HORIZON = 90


# --------------------------------------------------------------------------- configs
@dataclass
class ReinforceConfig:
    name: str
    lr: float = 1e-3
    gamma: float = 0.99
    episodes_per_batch: int = 10
    baseline: bool = True  # learned state-value baseline (REINFORCE with baseline)
    normalize_returns: bool = True
    entropy_coeff: float = 0.01
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"
    max_grad_norm: float = 1.0


@dataclass
class PPOConfig:
    name: str
    lr: float = 3e-4
    gamma: float = 0.99
    lmbda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    critic_coeff: float = 0.5
    epochs: int = 10
    minibatch_size: int = 256
    frames_per_batch: int = 2048
    normalize_advantage: bool = True
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"
    max_grad_norm: float = 0.5


@dataclass
class A2CConfig:
    name: str
    lr: float = 7e-4
    gamma: float = 0.99
    lmbda: float = 1.0  # 1.0 = Monte-Carlo-like advantage, <1 = GAE
    entropy_coeff: float = 0.01
    critic_coeff: float = 0.5
    frames_per_batch: int = 64  # n-step rollout length per update
    normalize_advantage: bool = False
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"
    max_grad_norm: float = 0.5


SWEEPS = {
    "reinforce": [
        ReinforceConfig("reinforce_01_baseline"),
        ReinforceConfig("reinforce_02_lr_3e-3", lr=3e-3),
        ReinforceConfig("reinforce_03_lr_3e-4", lr=3e-4),
        ReinforceConfig("reinforce_04_gamma_0.95", gamma=0.95),
        ReinforceConfig("reinforce_05_gamma_0.999", gamma=0.999),
        ReinforceConfig("reinforce_06_no_baseline", baseline=False),
        ReinforceConfig("reinforce_07_ep_per_batch_4", episodes_per_batch=4),
        ReinforceConfig("reinforce_08_ep_per_batch_25", episodes_per_batch=25),
        ReinforceConfig("reinforce_09_entropy_0", entropy_coeff=0.0),
        ReinforceConfig("reinforce_10_relu_256", hidden=[256, 256], activation="relu"),
    ],
    "ppo": [
        PPOConfig("ppo_01_baseline"),
        PPOConfig("ppo_02_lr_1e-3", lr=1e-3),
        PPOConfig("ppo_03_lr_1e-4", lr=1e-4),
        PPOConfig("ppo_04_clip_0.1", clip_epsilon=0.1),
        PPOConfig("ppo_05_clip_0.3", clip_epsilon=0.3),
        PPOConfig("ppo_06_gamma_0.95", gamma=0.95),
        PPOConfig("ppo_07_lambda_0.8", lmbda=0.8),
        PPOConfig("ppo_08_entropy_0.05", entropy_coeff=0.05),
        PPOConfig("ppo_09_epochs4_batch4096", epochs=4, frames_per_batch=4096),
        PPOConfig("ppo_10_relu_256", hidden=[256, 256], activation="relu"),
    ],
    "a2c": [
        A2CConfig("a2c_01_baseline"),
        A2CConfig("a2c_02_lr_2e-3", lr=2e-3),
        A2CConfig("a2c_03_lr_1e-4", lr=1e-4),
        A2CConfig("a2c_04_gamma_0.95", gamma=0.95),
        A2CConfig("a2c_05_gae_0.9", lmbda=0.9),
        A2CConfig("a2c_06_nstep_16", frames_per_batch=16),
        A2CConfig("a2c_07_nstep_256", frames_per_batch=256),
        A2CConfig("a2c_08_entropy_0.05", entropy_coeff=0.05),
        A2CConfig("a2c_09_critic_1.0_normadv", critic_coeff=1.0, normalize_advantage=True),
        A2CConfig("a2c_10_relu_256", hidden=[256, 256], activation="relu"),
    ],
}


# --------------------------------------------------------------------------- helpers
def build_actor_critic(env, cfg):
    obs_dim = env.observation_spec["observation"].shape[-1]
    n_actions = env.action_spec.shape[-1]
    actor_net = build_mlp(obs_dim, n_actions, cfg.hidden, cfg.activation)
    actor = ProbabilisticActor(
        TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"]),
        spec=env.action_spec,
        in_keys=["logits"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
    critic = ValueOperator(build_mlp(obs_dim, 1, cfg.hidden, cfg.activation), in_keys=["observation"])
    return actor_net, actor, critic


def reward_to_go(batch, gamma: float):
    """Discounted Monte-Carlo returns G_t for a [T] batch; mask for steps of unfinished episodes."""
    r = batch["next", "reward"].squeeze(-1)
    done = batch["next", "done"].squeeze(-1).float()
    g = torch.zeros_like(r)
    running = torch.zeros(())
    for t in reversed(range(r.shape[0])):
        running = r[t] + gamma * running * (1.0 - done[t])
        g[t] = running
    valid = torch.zeros_like(r, dtype=torch.bool)
    done_idx = torch.nonzero(done).flatten()
    if len(done_idx):
        valid[: done_idx[-1] + 1] = True  # drop the truncated tail after the last finished episode
    return g, valid


def entropy_of(actor, td) -> torch.Tensor:
    return actor.get_dist(td).entropy().mean()


# --------------------------------------------------------------------------- training
def train(algo: str, cfg, total_frames: int, seed: int = 0, eval_every: int = 10_000, verbose: bool = True) -> dict:
    set_seed(seed)
    env = make_env(seed)
    actor_net, actor, critic = build_actor_critic(env, cfg)

    if algo == "reinforce":
        loss_module = ReinforceLoss(actor, critic, loss_critic_type="smooth_l1")
        frames_per_batch = cfg.episodes_per_batch * HORIZON
        collector = Collector(env, actor, frames_per_batch=frames_per_batch, total_frames=total_frames, reset_at_each_iter=True)
    else:
        if algo == "ppo":
            loss_module = ClipPPOLoss(
                actor,
                critic,
                clip_epsilon=cfg.clip_epsilon,
                entropy_coeff=cfg.entropy_coeff,
                critic_coeff=cfg.critic_coeff,
                normalize_advantage=cfg.normalize_advantage,
                loss_critic_type="smooth_l1",
            )
        else:
            loss_module = A2CLoss(
                actor,
                critic,
                entropy_coeff=cfg.entropy_coeff,
                critic_coeff=cfg.critic_coeff,
                loss_critic_type="smooth_l1",
            )
        gae = GAE(gamma=cfg.gamma, lmbda=cfg.lmbda, value_network=critic, average_gae=False)
        collector = Collector(env, actor, frames_per_batch=cfg.frames_per_batch, total_frames=total_frames)

    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)
    logger = RunLogger(
        algo,
        cfg.name,
        ["frames", "train_return", "loss_actor", "loss_critic", "entropy", "eval_return", "eval_served_pct", "eval_failure_rate"],
    )
    ckpt_path = MODELS_DIR / "pg" / f"{cfg.name}.pt"
    frames, recent, next_eval = 0, [], eval_every
    best = {"eval_return": -float("inf")}
    stats = {"loss_actor": float("nan"), "loss_critic": float("nan"), "entropy": float("nan")}
    t0 = time.time()

    for batch in collector:
        frames += batch.numel()
        recent = (recent + episode_returns(batch))[-20:]

        if algo == "reinforce":
            g, valid = reward_to_go(batch, cfg.gamma)
            if valid.any():
                data = batch[valid]
                g = g[valid]
                with torch.no_grad():
                    v = critic(data.clone())["state_value"].squeeze(-1)
                adv = g - v if cfg.baseline else g
                if cfg.normalize_returns:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                data.set("advantage", adv.unsqueeze(-1))
                data.set("value_target", g.unsqueeze(-1))
                out = loss_module(data)
                ent = entropy_of(actor, data)
                loss = out["loss_actor"] - cfg.entropy_coeff * ent
                if cfg.baseline:
                    loss = loss + out["loss_value"]
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
                optim.step()
                stats = {"loss_actor": out["loss_actor"].item(), "loss_critic": out["loss_value"].item(), "entropy": ent.item()}

        elif algo == "ppo":
            for _ in range(cfg.epochs):
                with torch.no_grad():
                    gae(batch)
                perm = torch.randperm(batch.numel())
                for idx in perm.split(cfg.minibatch_size):
                    out = loss_module(batch[idx])
                    loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
                    optim.step()
            stats = {"loss_actor": out["loss_objective"].item(), "loss_critic": out["loss_critic"].item(), "entropy": out["entropy"].item()}

        else:  # a2c: one synchronous update per n-step rollout
            with torch.no_grad():
                gae(batch)
            if cfg.normalize_advantage and batch.numel() > 1:
                a = batch["advantage"]
                batch.set("advantage", (a - a.mean()) / (a.std() + 1e-8))
            out = loss_module(batch)
            loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
            optim.step()
            stats = {"loss_actor": out["loss_objective"].item(), "loss_critic": out["loss_critic"].item(), "entropy": out["entropy"].item()}

        collector.update_policy_weights_()

        if frames >= next_eval or frames >= total_frames:
            next_eval += eval_every
            metrics = evaluate(actor_net)
            row = {"frames": frames, "train_return": sum(recent) / max(len(recent), 1), **stats, **metrics}
            logger.log(row, verbose)
            if metrics["eval_return"] > best["eval_return"]:
                best = {**metrics, "frames": frames}
                save_checkpoint(ckpt_path, algo, actor_net, asdict(cfg), best)

    collector.shutdown()
    logger.close()
    summary = {"run": cfg.name, **asdict(cfg), **best, "minutes": (time.time() - t0) / 60, "seed": seed}
    append_summary(algo, summary)
    update_best(MODELS_DIR / "pg", algo, cfg.name, best, ckpt_path)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["reinforce", "ppo", "a2c", "all"], default="all")
    p.add_argument("--runs", type=int, nargs="*", help="indices (0-9) of sweep configs; default all")
    p.add_argument("--frames", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    algos = ["reinforce", "ppo", "a2c"] if args.algo == "all" else [args.algo]
    for algo in algos:
        sweep = SWEEPS[algo]
        for i in args.runs if args.runs else range(len(sweep)):
            cfg = sweep[i]
            print(f"\n=== {algo.upper()} run {i + 1}/{len(sweep)}: {cfg.name} ===")
            s = train(algo, cfg, args.frames, args.seed, verbose=not args.quiet)
            print(f"=== {cfg.name}: best eval return {s['eval_return']:.1f}, served {s['eval_served_pct']:.1f}% ===")


if __name__ == "__main__":
    main()
