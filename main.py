"""Water Point Maintenance Dispatch - entry point.

    uv run main.py                         # 3D simulation of the best trained agent (all algorithms compared)
    uv run main.py --algo ppo              # simulate a specific algorithm's best model (dqn|reinforce|ppo|a2c)
    uv run main.py --algo heuristic        # rule-based baseline; --algo random for a random agent
    uv run main.py --episodes 2 --seed 5   # number of episodes / starting seed
    uv run main.py --no-render             # terminal only
    uv run main.py train --algo all        # run all hyperparameter sweeps (40 runs)
    uv run main.py compare                 # evaluate best models + baselines, write plots to assets/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BEST_FILES = {
    "dqn": ROOT / "models" / "dqn" / "best_dqn",
    "reinforce": ROOT / "models" / "pg" / "best_reinforce",
    "ppo": ROOT / "models" / "pg" / "best_ppo",
    "a2c": ROOT / "models" / "pg" / "best_a2c",
}

C = {"g": "\033[92m", "y": "\033[93m", "r": "\033[91m", "b": "\033[94m", "dim": "\033[2m", "bold": "\033[1m", "end": "\033[0m"}


def pick_best_algo() -> str | None:
    best, best_ret = None, -float("inf")
    for algo, stem in BEST_FILES.items():
        meta = stem.with_suffix(".json")
        if meta.exists() and stem.with_suffix(".pt").exists():
            ret = json.loads(meta.read_text())["eval_return"]
            print(f"  {algo:<10} best run {json.loads(meta.read_text())['run']:<28} eval return {ret:8.1f}")
            if ret > best_ret:
                best, best_ret = algo, ret
    return best


def make_agent(algo: str, env):
    if algo == "random":
        return lambda obs: env.action_space.sample(), "random agent"
    if algo == "heuristic":
        from environment import heuristic_action

        return lambda obs: heuristic_action(env), "rule-based heuristic"
    import torch

    from training.common import load_policy_net

    path = BEST_FILES[algo].with_suffix(".pt")
    if not path.exists():
        sys.exit(f"No trained model at {path}. Train first: uv run main.py train --algo {algo}")
    net, ckpt = load_policy_net(path)
    print(f"Loaded {path.relative_to(ROOT)}  (run: {ckpt['config']['name']}, eval return {ckpt['metrics']['eval_return']:.1f})")

    def act(obs):
        with torch.no_grad():
            return int(net(torch.as_tensor(obs).unsqueeze(0)).argmax(-1).item())

    return act, f"{algo.upper()} - {ckpt['config']['name']}"


def status_strip(env) -> str:
    sym = {0: C["g"] + "●" + C["end"], 1: C["y"] + "●" + C["end"], 2: C["r"] + "●" + C["end"]}
    return "".join(sym[int(s)] for s in env.status)


def demo(args):
    from environment import WaterDispatchEnv

    algo = args.algo
    if algo == "best":
        print("Comparing best models of each algorithm:")
        algo = pick_best_algo()
        if algo is None:
            print("No trained models found - running the rule-based heuristic instead.")
            print("Train with: uv run main.py train --algo all")
            algo = "heuristic"
        else:
            print(f"-> best performing algorithm: {algo.upper()}\n")

    env = WaterDispatchEnv(render_mode=None if args.no_render else "human")
    act, label = make_agent(algo, env)
    if not args.no_render:
        env.reset(seed=args.seed)
        env.render()
        env._renderer.caption_extra = f"Agent: {label}"

    all_returns, all_served = [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        print(f"\n{C['bold']}=== Episode {ep + 1}/{args.episodes}  |  {label}  |  seed {args.seed + ep} ==={C['end']}")
        print(f"{C['dim']}start: {info['n_functional']} functional, {info['n_needs_repair']} need repair, "
              f"{info['n_non_functional']} non functional, {info['pct_population_served']:.0f}% of people served{C['end']}")
        print(f"{C['dim']}{'day':>4} {'action':<17} {'loc':<6} {'reward':>8} {'return':>8} {'fuel':>6} {'budget':>7} kits  points       event{C['end']}")
        done = False
        while not done:
            a = act(obs)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            rc = C["g"] if r >= 0 else C["r"]
            print(
                f"{info['day']:>4} {env.action_name(a):<17} {info['location']:<6} {rc}{r:>+8.2f}{C['end']} "
                f"{env.episode_return:>+8.1f} {info['fuel']:>6.1f} {info['budget']:>7.0f} {info['minor_kits']}/{info['major_kits']}   "
                f"{status_strip(env)}   {C['dim']}{env.last_event}{C['end']}"
            )
        served = 100 * info["people_days_served"] / (env.pop.sum() * env.cfg.horizon_days)
        all_returns.append(env.episode_return)
        all_served.append(served)
        end = "STRANDED (out of fuel)" if info["stranded"] else ("BUDGET EXHAUSTED" if env.budget <= 0 else "90-day horizon reached")
        print(f"{C['b']}Episode {ep + 1} finished: {end}. Return {env.episode_return:.1f} | "
              f"people-days with water {served:.1f}% | repairs {info['repairs_done']} | km {info['km_driven']:.0f} | "
              f"budget left ${info['budget']:.0f}{C['end']}")
    print(f"\n{C['bold']}Summary over {args.episodes} episodes: mean return {np.mean(all_returns):.1f}, "
          f"mean people-days served {np.mean(all_served):.1f}%{C['end']}")
    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    p.add_argument("--algo", default="best", choices=["best", "dqn", "reinforce", "ppo", "a2c", "heuristic", "random"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--no-render", action="store_true")

    t = sub.add_parser("train", help="run hyperparameter sweeps")
    t.add_argument("--algo", default="all", choices=["dqn", "reinforce", "ppo", "a2c", "all"])
    t.add_argument("--frames", type=int, default=300_000)
    t.add_argument("--runs", type=int, nargs="*")

    c = sub.add_parser("compare", help="evaluate best models and baselines, make report plots")
    c.add_argument("--episodes", type=int, default=100)

    args = p.parse_args()
    if args.cmd == "train":
        if args.algo in ("dqn", "all"):
            from training import dqn_training

            for i in args.runs or range(len(dqn_training.SWEEP)):
                dqn_training.train(dqn_training.SWEEP[i], args.frames)
        if args.algo != "dqn":
            from training import pg_training

            for algo in ["reinforce", "ppo", "a2c"] if args.algo == "all" else [args.algo]:
                for i in args.runs or range(len(pg_training.SWEEPS[algo])):
                    print(f"\n=== {algo.upper()} {pg_training.SWEEPS[algo][i].name} ===")
                    pg_training.train(algo, pg_training.SWEEPS[algo][i], args.frames)
    elif args.cmd == "compare":
        from training.evaluate import compare

        compare(args.episodes)
    else:
        demo(args)


if __name__ == "__main__":
    main()
