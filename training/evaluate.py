"""Compare the best model of each algorithm against baselines and produce report figures.

    uv run main.py compare            (or: uv run python -m training.evaluate)

Outputs
    logs/comparison.csv                   per-agent evaluation table
    assets/learning_curves.png            eval return vs frames, best run per algorithm
    assets/sweep_<algo>.png               all 10 hyperparameter runs per algorithm
    assets/sweep_summary.png              best eval return of every run, per algorithm
    assets/training_diagnostics.png       DQN loss / Q, PG policy entropy
    assets/return_distribution.png        evaluation return distribution per agent
"""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from environment import WaterDispatchEnv, heuristic_action  # noqa: E402
from training.common import LOGS_DIR, MODELS_DIR, ROOT, load_policy_net  # noqa: E402

ASSETS = ROOT / "assets"
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C", "heuristic": "Heuristic", "random": "Random"}
# categorical slots 1-4 of the validated reference palette (fixed order); baselines in neutral ink
COLOR = {"dqn": "#2a78d6", "reinforce": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100", "heuristic": "#52514e", "random": "#a3a29c"}
BEST = {
    "dqn": MODELS_DIR / "dqn" / "best_dqn",
    "reinforce": MODELS_DIR / "pg" / "best_reinforce",
    "ppo": MODELS_DIR / "pg" / "best_ppo",
    "a2c": MODELS_DIR / "pg" / "best_a2c",
}
TEST_SEED = 200_000  # unseen during training and model selection

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#2b2b2a",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def run_agent(act, env, n: int) -> dict:
    rets, served, fails, repairs, km = [], [], [], [], []
    for i in range(n):
        obs, info = env.reset(seed=TEST_SEED + i)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(act(obs))
            done = term or trunc
        rets.append(env.episode_return)
        served.append(100 * info["people_days_served"] / (env.pop.sum() * env.cfg.horizon_days))
        fails.append(float(info["stranded"] or env.budget <= 0))
        repairs.append(info["repairs_done"])
        km.append(info["km_driven"])
    return {"returns": rets, "served": served, "fails": fails, "repairs": repairs, "km": km}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def col(rows, key):
    return np.array([float(r[key]) if r[key] not in ("", "nan") else np.nan for r in rows])


def compare(n_episodes: int = 100):
    ASSETS.mkdir(exist_ok=True)
    env = WaterDispatchEnv()
    results = {}
    for algo in ALGOS:
        if BEST[algo].with_suffix(".pt").exists():
            net, _ = load_policy_net(BEST[algo].with_suffix(".pt"))

            def act(obs, net=net):
                with torch.no_grad():
                    return int(net(torch.as_tensor(obs).unsqueeze(0)).argmax(-1).item())

            results[algo] = run_agent(act, env, n_episodes)
    results["heuristic"] = run_agent(lambda o: heuristic_action(env), env, n_episodes)
    results["random"] = run_agent(lambda o: env.action_space.sample(), env, n_episodes)

    # ---------------- table
    rows = []
    print(f"\nEvaluation on {n_episodes} unseen episodes (seeds {TEST_SEED}+):")
    print(f"{'agent':<11}{'best run':<28}{'return':>16}{'served %':>10}{'fail %':>8}{'repairs':>9}{'km':>7}")
    for name, r in results.items():
        run = json.loads(BEST[name].with_suffix(".json").read_text())["run"] if name in BEST else "-"
        row = {
            "agent": LABEL[name],
            "best_run": run,
            "mean_return": np.mean(r["returns"]),
            "std_return": np.std(r["returns"]),
            "served_pct": np.mean(r["served"]),
            "failure_pct": 100 * np.mean(r["fails"]),
            "repairs": np.mean(r["repairs"]),
            "km_driven": np.mean(r["km"]),
        }
        rows.append(row)
        print(
            f"{row['agent']:<11}{run:<28}{row['mean_return']:>8.1f} ± {row['std_return']:<5.1f}"
            f"{row['served_pct']:>10.1f}{row['failure_pct']:>8.1f}{row['repairs']:>9.1f}{row['km_driven']:>7.0f}"
        )
    with open(LOGS_DIR / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------------- return distribution
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = list(results)
    bp = ax.boxplot([results[n]["returns"] for n in names], patch_artist=True, widths=0.55, showfliers=False)
    for patch, n in zip(bp["boxes"], names):
        patch.set(facecolor=COLOR[n], alpha=0.85, edgecolor="white", linewidth=2)
    for med in bp["medians"]:
        med.set(color="#0b0b0b", linewidth=1.5)
    ax.set_xticks(range(1, len(names) + 1), [LABEL[n] for n in names])
    ax.set_ylabel("Episode return")
    ax.set_title(f"Return on {n_episodes} unseen episodes", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "return_distribution.png")
    plt.close(fig)

    plot_training_curves(heuristic_return=float(np.mean(results["heuristic"]["returns"])))
    print(f"\nWrote logs/comparison.csv and figures in {ASSETS.relative_to(ROOT)}/")


def plot_training_curves(heuristic_return: float | None = None):
    available = [a for a in ALGOS if (LOGS_DIR / a / "summary.csv").exists()]
    if not available:
        return

    # learning curves of the best run per algorithm
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for algo in available:
        best_run = json.loads(BEST[algo].with_suffix(".json").read_text())["run"]
        rows = read_csv(LOGS_DIR / algo / f"{best_run}.csv")
        x, y = col(rows, "frames") / 1000, col(rows, "eval_return")
        ax.plot(x, y, color=COLOR[algo], label=f"{LABEL[algo]} ({best_run})")
        ax.annotate(LABEL[algo], (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color="#2b2b2a")
    if heuristic_return is not None:
        ax.axhline(heuristic_return, color=COLOR["heuristic"], linestyle="--", linewidth=1.5, label="Heuristic baseline")
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Greedy evaluation return")
    ax.set_title("Learning curves - best configuration per algorithm", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "learning_curves.png")
    plt.close(fig)

    # sweeps: every run per algorithm (single hue, best run emphasised)
    for algo in available:
        summary = read_csv(LOGS_DIR / algo / "summary.csv")
        best_run = json.loads(BEST[algo].with_suffix(".json").read_text())["run"]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for s in summary:
            path = LOGS_DIR / algo / f"{s['run']}.csv"
            if not path.exists():
                continue
            rows = read_csv(path)
            is_best = s["run"] == best_run
            ax.plot(
                col(rows, "frames") / 1000,
                col(rows, "eval_return"),
                color=COLOR[algo] if is_best else "#b9b8b2",
                linewidth=2.4 if is_best else 1.1,
                zorder=3 if is_best else 2,
                label=f"{s['run']} (best)" if is_best else None,
            )
        ax.set_xlabel("Environment steps (thousands)")
        ax.set_ylabel("Greedy evaluation return")
        ax.set_title(f"{LABEL[algo]} - {len(summary)} hyperparameter configurations (grey) and best (colour)", loc="left", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(ASSETS / f"sweep_{algo}.png")
        plt.close(fig)

    # sweep summary: best eval return of each run (small multiples, one hue each)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, algo in zip(axes, available):
        summary = sorted(read_csv(LOGS_DIR / algo / "summary.csv"), key=lambda r: r["run"])
        vals = [float(r["eval_return"]) for r in summary]
        labels = [r["run"].split("_", 2)[-1] for r in summary]
        y = np.arange(len(vals))
        ax.barh(y, vals, color=COLOR[algo], height=0.7, edgecolor="white", linewidth=2)
        ax.set_yticks(y, labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(LABEL[algo], loc="left", fontsize=10)
        ax.axvline(0, color="#8a8984", linewidth=0.8)
        ax.grid(axis="y", visible=False)
    fig.supxlabel("Best greedy evaluation return", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "sweep_summary.png")
    plt.close(fig)

    # diagnostics: DQN loss, PG entropy (two separate panels, one axis each)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    if "dqn" in available:
        best_run = json.loads(BEST["dqn"].with_suffix(".json").read_text())["run"]
        rows = read_csv(LOGS_DIR / "dqn" / f"{best_run}.csv")
        a1.plot(col(rows, "frames") / 1000, col(rows, "loss"), color=COLOR["dqn"])
        a1.set_title(f"DQN TD loss ({best_run})", loc="left", fontsize=10)
        a1.set_xlabel("Steps (thousands)")
    for algo in [a for a in ("reinforce", "ppo", "a2c") if a in available]:
        best_run = json.loads(BEST[algo].with_suffix(".json").read_text())["run"]
        rows = read_csv(LOGS_DIR / algo / f"{best_run}.csv")
        a2.plot(col(rows, "frames") / 1000, col(rows, "entropy"), color=COLOR[algo], label=LABEL[algo])
    a2.set_title("Policy entropy (best runs)", loc="left", fontsize=10)
    a2.set_xlabel("Steps (thousands)")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "training_diagnostics.png")
    plt.close(fig)


if __name__ == "__main__":
    compare()
