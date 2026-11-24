"""Build the assignment report PDF from the training logs and figures.

    uv run --with reportlab python report/build_report.py
"""

from __future__ import annotations

import json

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer

from pdf_style import (
    ASSETS,
    GITHUB,
    ROOT,
    STUDENT,
    VIDEO,
    P,
    best_run,
    bullets,
    callout,
    figure,
    read_comparison,
    read_metrics,
    read_summary,
    styles,
    table,
)

OUT = ROOT / "report" / "Mahamat_RL_Summative_Report.pdf"
st = styles(9.6)
W = A4[0] - 3.6 * cm  # text width


def f1(x):
    return f"{float(x):.1f}"


def hp_table(algo, columns, getters, widths):
    rows = [columns]
    runs = read_summary(algo)
    best = best_run(algo)
    hi = None
    for i, r in enumerate(runs, start=1):
        rows.append([g(r) for g in getters])
        if r["run"] == best:
            hi = i
    return table(rows, [w * cm for w in widths], st, highlight_row=hi)


def short(r):
    return r["run"].split("_", 2)[-1].replace("_", " ")


def mean_reward(r):
    return f"<b>{f1(r['eval_return'])}</b>"


def served(r):
    return f"{float(r['eval_served_pct']):.0f}%"


def fail(r):
    return f"{100 * float(r['eval_failure_rate']):.0f}%"


def hidden(r):
    return "×".join(str(h) for h in json.loads(r["hidden"])) + f" {r['activation']}"


def build():
    comp = read_comparison()
    met = read_metrics()
    s = []

    # ------------------------------------------------------------ title
    s.append(P("Reinforcement Learning Summative Assignment Report", st["title"]))
    s.append(P(f"<b>Student:</b> {STUDENT}", st["subtitle"]))
    s.append(P(f"<b>Video recording:</b> {VIDEO}", st["subtitle"]))
    s.append(P(f'<b>GitHub repository:</b> <link href="{GITHUB}" color="#2a78d6">{GITHUB}</link>', st["subtitle"]))
    s.append(Spacer(1, 4))

    # ------------------------------------------------------------ 1 overview
    s.append(P("1. Project Overview", st["h1"]))
    s.append(P(
        "My capstone classifies Tanzanian rural water points (the <i>Pump it Up</i> dataset) as functional, needing repair "
        "or non-functional. A classifier only <i>predicts</i> failures; district water offices still have to decide "
        "<i>where to send their single maintenance crew next</i>. This project turns that decision into a reinforcement "
        "learning problem. A custom Gymnasium environment simulates one crew with a pickup truck, limited fuel, a repair "
        "budget and spare-part kits, maintaining 10 water points over a 90-day horizon. Pumps degrade every day according to "
        "a failure-risk score built from Pump-it-Up features, and the crew only knows the status it last observed. Four "
        "algorithms were implemented with <b>TorchRL</b> on the identical environment: DQN (value-based) and REINFORCE, PPO "
        "and A2C (policy gradient). Each was tuned over 10 hyperparameter configurations (40 runs in total, 300k steps each). "
        f"The best agent (DQN) keeps water flowing for <b>{f1(comp['DQN']['served_pct'])}%</b> of people-days on unseen "
        f"episodes, beating a hand-written dispatch heuristic ({f1(comp['Heuristic']['served_pct'])}%). PPO generalises best "
        "to new district layouts.", st["body"]))

    # ------------------------------------------------------------ 2 environment
    s.append(P("2. Environment Description", st["h1"]))
    s.append(P("2.1 Agent(s)", st["h2"]))
    s.append(P(
        "A single agent represents the district maintenance crew and its truck. Each step is one working day. The agent "
        "is partially informed: it sees its own resources exactly, but for each water point only the <i>last known</i> "
        "status and how many days ago it was observed. True statuses change silently, so the agent must decide when "
        "information is stale enough to justify an inspection trip. It also has to plan refuelling, because running out of "
        "fuel away from the depot ends the episode.", st["body"]))

    s.append(P("2.2 Action Space", st["h2"]))
    s.append(P("Discrete(15). Every action maps to something a real maintenance crew does in a day:", st["body"]))
    s.append(table([
        ["Action", "Meaning", "Effect / constraint"],
        ["0-9", "Drive to water point k", "Uses 0.25 L/km of fuel; −0.02 reward per km; same-location = invalid"],
        ["10", "Return to depot", "Refuel to 60 L ($1.50/L) and restock kits (minor $40, major $250)"],
        ["11", "Inspect", "Reveals the true status of the current point; invalid at the depot"],
        ["12", "Minor repair", "1 minor kit + $20 labour; fixes <i>needs repair</i> only"],
        ["13", "Major repair", "1 major kit + $80 labour; fixes <i>needs repair</i> and <i>non-functional</i>"],
        ["14", "Wait", "Stand by for the day"],
    ], [1.4 * cm, 3.6 * cm, W - 5.0 * cm], st))

    s.append(P("2.3 Observation Space", st["h2"]))
    s.append(P(
        "Box(86,), all values normalised to [0, 1]: crew location one-hot (11 = 10 points + depot); fuel, budget, minor "
        "kits and major kits as fractions of capacity; day/90; and for each of the 10 points: last known status one-hot "
        "(3), days since last observation, population served, failure-risk score, and distance from the crew. The risk score "
        "is a logistic function of pump age, extraction type and payment type, with coefficients chosen to match the "
        "direction of effects in the capstone. Predictions from the real classifier can be loaded from a CSV instead.",
        st["body"]))

    s.append(P("2.4 Reward Structure", st["h2"]))
    s.append(P("Populations are expressed in thousands (pop<sub>k</sub>). The reward for one day is:", st["body"]))
    s.append(P(
        "r<sub>t</sub> = 10·pop<sub>k</sub>·[restored] (×0.5 if it only needed repair) − 0.5·Σ<sub>k</sub> pop<sub>k</sub>·down<sub>k</sub> "
        "− 0.02·km − 1·[invalid] − 1.5·[wasted repair] + 0.5·risk<sub>k</sub>·[inspect stale point]", st["code"]))
    s.append(P(
        "Here down<sub>k</sub> = 1 for non-functional points and 0.3 for points that need repair. A <i>stale</i> point "
        "is one that has not been observed for 7 or more days. Running out of fuel costs −30 and bankruptcy −10. On early "
        "termination the agent is also charged the current downtime for every remaining day, so ending an episode is "
        "never cheaper than continuing. The dominant term is the daily downtime penalty, weighted by population, which "
        "directly encodes the mission: <i>maximise people-days with water</i>.", st["body"]))

    s.append(P("2.5 Start State and Terminal Conditions", st["h2"]))
    s.append(P(
        "<b>Start:</b> the crew is at the central depot with full fuel (60 L), full kits (4 minor, 2 major) and a $6,000 budget. "
        "Point statuses are sampled from the Pump it Up class distribution (54.3% functional, 7.3% need repair, 38.4% "
        "non-functional), and the survey results are 0-30 days old. <b>Terminal:</b> the 90-day horizon is reached "
        "(truncation), the truck runs out of fuel away from the depot, or the budget is exhausted.", st["body"]))

    s.append(P("2.6 Environment Visualization", st["h2"]))
    s.append(figure(ASSETS / "env_screenshot.png", 15.5,
                    "Figure 1 - Interactive 3D OpenGL simulation (pygame + PyOpenGL). Tank band = true status, floating diamond = "
                    "crew's belief (fades with age), village size = population. HUD: resources, last action, reward, minimap.", st))
    s.append(P(
        "The renderer draws rolling savanna terrain, roads, villages whose hut count scales with population, and "
        "elevated tanks that show their true status, with flashing beacons on broken points and villagers queuing at "
        "working taps. The truck drives with a dust trail. Inspections and repairs trigger ring and spark effects. The user "
        "can orbit, pan, zoom and tilt the camera, follow the truck, and click any water point to open a details panel "
        "with its features, its true status and the crew's belief (Figure 2).", st["body"]))
    s.append(figure(ASSETS / "env_closeup.png", 13.5, "Figure 2 - Close-up view with the details panel of a selected water point.", st))

    # ------------------------------------------------------------ 3 system design
    s.append(P("3. System Analysis and Design", st["h1"]))
    s.append(P("3.1 Deep Q-Network (DQN)", st["h2"]))
    s.append(P(
        "A TorchRL <font face='Courier'>QValueActor</font> wraps an MLP (86 → 128 → 128 → 15, ReLU) that outputs one "
        "Q-value per action. Exploration is ε-greedy (<font face='Courier'>EGreedyModule</font>, ε annealed from 1.0 "
        "to 0.05 over 40% of training). Transitions go into a 50k <font face='Courier'>ReplayBuffer</font> "
        "(<font face='Courier'>LazyTensorStorage</font>), and minibatches are sampled uniformly to break temporal correlation. "
        "<font face='Courier'>DQNLoss</font> (smooth-L1, TD(0) target) uses a delayed <b>target network</b> that is updated "
        "every 500 gradient steps (<font face='Courier'>HardUpdate</font>) or by Polyak averaging "
        "(<font face='Courier'>SoftUpdate</font>). Double DQN was tested as a variant. The agent does 50 gradient steps per "
        "200 collected frames after a 5k-frame random warm-up, with gradient clipping at 10.", st["body"]))
    s.append(P("3.2 Policy Gradient Methods", st["h2"]))
    s.append(P(
        "All three share the same architecture: a <font face='Courier'>ProbabilisticActor</font> (MLP 86 → 128 → 128 → 15, tanh, "
        "one-hot categorical distribution over the logits) and a <font face='Courier'>ValueOperator</font> critic "
        "(86 → 128 → 128 → 1).", st["body"]))
    s.extend(bullets([
        "<b>REINFORCE</b> (<font face='Courier'>ReinforceLoss</font>): Monte-Carlo policy gradient. Each update collects "
        "whole episodes (10 by default), computes discounted returns-to-go G<sub>t</sub> (truncated tails are masked out), "
        "and uses the advantage G<sub>t</sub> − V(s<sub>t</sub>) with a learned baseline, normalised, plus an entropy bonus.",
        "<b>A2C</b> (<font face='Courier'>A2CLoss</font> + <font face='Courier'>GAE</font>): synchronous advantage "
        "actor-critic. One gradient step per 64-step rollout, with advantages from GAE (λ = 1 by default). "
        "Loss = policy + 0.5·critic − 0.01·entropy.",
        "<b>PPO</b> (<font face='Courier'>ClipPPOLoss</font> + <font face='Courier'>GAE</font>): 2048-step rollouts, GAE "
        "(γ = 0.99, λ = 0.95), 10 epochs of 256-sample minibatches, clipped ratio ε = 0.2, normalised advantages, and "
        "gradient clipping at 0.5.",
    ], st))
    s.append(P(
        "<b>Fair comparison:</b> every run uses the same TorchRL-wrapped environment "
        "(<font face='Courier'>GymWrapper</font> + <font face='Courier'>StepCounter</font>, "
        "<font face='Courier'>RewardSum</font>, <font face='Courier'>RewardScaling</font> 0.1), the same 300k-step budget and "
        "seed 0. A greedy evaluation runs every 10k steps on 20 held-out seeds, and the best checkpoint is kept. Final "
        "numbers come from 100 further unseen seeds.", st["body"]))

    # ------------------------------------------------------------ 4 implementation (hyperparameter tables)
    s.append(P("4. Implementation", st["h1"]))
    s.append(P(
        "Each table lists the 10 runs. <i>Mean reward</i> is the best greedy evaluation return over 20 held-out episodes "
        "(higher is better; the heuristic baseline scores ≈ 15 and random ≈ −233). The highlighted row is the selected model. "
        "Settings not listed are the defaults described in Section 3.", st["body"]))

    s.append(P("4.1 DQN", st["h2"]))
    s.append(hp_table(
        "dqn",
        ["Run", "Learning rate", "Gamma", "Replay buffer", "Batch size", "Exploration strategy", "Target / variant", "Served", "Mean reward"],
        [short, lambda r: r["lr"], lambda r: r["gamma"], lambda r: f"{int(r['buffer_size']) // 1000}k", lambda r: r["batch_size"],
         lambda r: f"ε 1.0→{r['eps_end']} over {float(r['eps_decay_frac']) * 100:.0f}%",
         lambda r: ("double, " if r["double_dqn"] == "True" else "") + (f"hard/{r['target_interval']}" if r["target_update"] == "hard" else f"soft τ={r['tau']}")
         + (", 256×256" if "256" in r["hidden"] else ""),
         served, mean_reward],
        [2.6, 1.4, 1.35, 1.4, 1.2, 2.9, 2.6, 1.2, 1.6]))
    s.append(Spacer(1, 4))
    s.append(P("4.2 REINFORCE", st["h2"]))
    s.append(hp_table(
        "reinforce",
        ["Run", "Learning rate", "Gamma", "Episodes / update", "Baseline", "Entropy coef.", "Network", "Fail", "Served", "Mean reward"],
        [short, lambda r: r["lr"], lambda r: r["gamma"], lambda r: r["episodes_per_batch"], lambda r: "yes" if r["baseline"] == "True" else "no",
         lambda r: r["entropy_coeff"], hidden, fail, served, mean_reward],
        [2.7, 1.4, 1.35, 1.6, 1.55, 1.4, 2.2, 1.0, 1.2, 1.6]))
    s.append(Spacer(1, 4))
    s.append(P("4.3 PPO", st["h2"]))
    s.append(hp_table(
        "ppo",
        ["Run", "Learning rate", "Gamma", "GAE λ", "Clip ε", "Entropy coef.", "Epochs × rollout", "Network", "Served", "Mean reward"],
        [short, lambda r: r["lr"], lambda r: r["gamma"], lambda r: r["lmbda"], lambda r: r["clip_epsilon"], lambda r: r["entropy_coeff"],
         lambda r: f"{r['epochs']} × {r['frames_per_batch']}", hidden, served, mean_reward],
        [2.7, 1.4, 1.35, 1.1, 1.1, 1.4, 2.0, 2.2, 1.2, 1.6]))
    s.append(Spacer(1, 4))
    s.append(P("4.4 Actor-Critic (A2C)", st["h2"]))
    s.append(hp_table(
        "a2c",
        ["Run", "Learning rate", "Gamma", "GAE λ", "n-step rollout", "Entropy coef.", "Critic coef. / adv. norm", "Network", "Fail", "Mean reward"],
        [short, lambda r: r["lr"], lambda r: r["gamma"], lambda r: r["lmbda"], lambda r: r["frames_per_batch"], lambda r: r["entropy_coeff"],
         lambda r: f"{r['critic_coeff']} / {'yes' if r['normalize_advantage'] == 'True' else 'no'}", hidden, fail, mean_reward],
        [2.7, 1.4, 1.35, 1.1, 1.5, 1.4, 2.2, 2.0, 1.0, 1.6]))
    s.append(Spacer(1, 4))
    s.append(P("4.5 Behaviour observed during tuning", st["h2"]))
    s.extend(bullets([
        "<b>DQN</b> was robust: all 10 runs beat the heuristic baseline. The <b>learning rate mattered most</b>: 1e-4 gave "
        "the best score (53.8), while 1e-3 gave noisy TD targets and the worst run (10.9). γ = 0.95 was almost as good "
        "as 0.99 because the daily downtime penalty is dense, so long-horizon credit is not essential. Faster ε-decay "
        "(15%) helped, and the larger replay buffer and batch (100k/256) slowed learning down.",
        "<b>REINFORCE</b> was the most sensitive. Its high-variance Monte-Carlo gradients made three runs (lr 3e-4, γ = 0.999, "
        "25 episodes per update) collapse into an always-<i>wait</i> policy that never improved after the first evaluation. "
        "Lowering γ to 0.95 shortened the credit horizon and cut the variance, giving the best run (−117.9). The learned "
        "baseline beat no baseline (−187.5 vs −225.4), and smaller, more frequent updates (4 episodes) helped.",
        "<b>PPO</b> improved steadily in every run. A lower GAE λ (0.8) and a looser clip (0.3) gave the best results "
        "(20.0 / 15.2): with 15 actions and sparse repair rewards, slightly more aggressive updates helped. A tight clip "
        "(0.1), a large entropy bonus (0.05) or too few epochs left the policy under-trained, and lr 1e-4 was too slow.",
        "<b>A2C</b> showed <b>reward hacking</b>. Several configurations (n-step 16, lr 2e-3, normalised advantage, 256-unit "
        "ReLU) learned to repair a few points and then <i>deliberately strand the truck</i> (100% failure rate, only 9-19% "
        "of people-days served). This is because the early-termination penalty charges current downtime but not the future "
        "failures the agent avoids by ending early. γ = 0.95 (−32.7) and GAE λ = 0.9 avoided this and learned a genuine "
        "dispatch policy.",
    ], st))

    # ------------------------------------------------------------ 5 results
    s.append(P("5. Results Discussion", st["h1"]))
    s.append(P(
        "Table 5 evaluates the selected model of each algorithm on 100 new episodes (unseen seeds) against two baselines: "
        "a rule-based dispatcher that goes to the most-populated broken point, inspects stale points and refuels when low, "
        "and a uniformly random agent.", st["body"]))
    rows = [["Agent", "Selected run", "Mean return ± std", "People-days served", "Episodes failed", "Repairs / episode", "km / episode"]]
    for a in ["DQN", "PPO", "A2C", "REINFORCE", "Heuristic", "Random"]:
        r = comp[a]
        rows.append([f"<b>{a}</b>", r["best_run"], f"{f1(r['mean_return'])} ± {f1(r['std_return'])}", f"{f1(r['served_pct'])}%",
                     f"{float(r['failure_pct']):.0f}%", f1(r["repairs"]), f"{float(r['km_driven']):.0f}"])
    s.append(table(rows, [2.0 * cm, 3.6 * cm, 2.8 * cm, 2.4 * cm, 2.0 * cm, 2.1 * cm, 1.9 * cm], st, highlight_row=1))
    s.append(P("Table 5 - Final evaluation on 100 unseen episodes.", st["caption"]))

    s.append(P("5.1 Cumulative Rewards", st["h2"]))
    s.append(figure(ASSETS / "learning_curves.png", 13.2, "Figure 3 - Greedy evaluation return during training, best run per algorithm.", st))
    s.append(P(
        f"DQN reaches the heuristic's level after ~150k steps and finishes highest ({f1(comp['DQN']['mean_return'])}). "
        f"Compared with the heuristic it performs ~1.4 more repairs per episode and drives 24% fewer km, because it learned "
        "to chain nearby repairs instead of returning to the depot. PPO crosses the heuristic line at ~220k steps. A2C and "
        "REINFORCE improve throughout but remain below it within the 300k-step budget. REINFORCE's curve shows the largest "
        "swings, the signature of high-variance Monte-Carlo gradients.", st["body"]))
    s.append(figure(ASSETS / "return_distribution.png", 11.5, "Figure 4 - Distribution of episode returns on 100 unseen episodes.", st))

    s.append(P("5.2 Training Stability", st["h2"]))
    s.append(figure(ASSETS / "training_diagnostics.png", 14.5,
                    "Figure 5 - Left: DQN TD loss (smooth-L1). Right: policy entropy of the best PG runs.", st))
    s.append(P(
        "The DQN TD loss falls quickly, with a single spike at ~60k steps (the end of the random warm-up and "
        "ε-annealing, when the data distribution shifts), and then stays small and bounded with no divergence. "
        f"Over the last 10 evaluations the spread of returns was: DQN σ = {met['dqn']['final10_std']:.1f}, PPO σ = {met['ppo']['final10_std']:.1f}, "
        f"A2C σ = {met['a2c']['final10_std']:.1f}, REINFORCE σ = {met['reinforce']['final10_std']:.1f}. PPO's entropy decreases "
        "smoothly and furthest (2.6 → 0.7), a controlled move from exploration to a confident policy. A2C's falls more noisily "
        "to ~1.2, and REINFORCE's stays near 2.0, meaning its weak, noisy gradient never let the policy commit. PPO's large "
        "test-time standard deviation (±64.5) comes from a few extreme outlier episodes; its median and interquartile range "
        "(Figure 4) are as tight as the heuristic's.", st["body"]))
    s.append(P("5.3 Episodes to Converge", st["h2"]))
    rows = [["Algorithm", "Steps to 90% of final improvement", "≈ Episodes (90 days)", "Final-10 eval mean ± σ"]]
    for a, lab in [("dqn", "DQN"), ("ppo", "PPO"), ("a2c", "A2C"), ("reinforce", "REINFORCE")]:
        m = met[a]
        rows.append([lab, f"{m['conv_frames'] // 1000}k", f"{m['conv_episodes']:,}", f"{m['final10_mean']:.1f} ± {m['final10_std']:.1f}"])
    s.append(table(rows, [3 * cm, 5 * cm, 4 * cm, 4.5 * cm], st))
    s.append(Spacer(1, 4))
    s.append(P(
        "DQN converges fastest (~1,700 episodes). Off-policy replay lets it reuse every transition ~30 times, while the "
        "on-policy methods throw data away after one (A2C, REINFORCE) or ten (PPO) passes. PPO needed ~2,300 episodes, and "
        "A2C and REINFORCE were still improving at the end of the budget, so they had not fully converged.", st["body"]))

    s.append(P("5.4 Generalization", st["h2"]))
    rows = [["Agent", "Unseen days, same district: return / served", "New random district layouts: return / served"]]
    for a, lab in [("dqn", "DQN"), ("ppo", "PPO"), ("a2c", "A2C"), ("reinforce", "REINFORCE"), ("heuristic", "Heuristic"), ("random", "Random")]:
        c = comp[lab]
        m = met[a]
        rows.append([lab, f"{f1(c['mean_return'])} / {f1(c['served_pct'])}%", f"{m['newmap_return']:.1f} / {m['newmap_served']:.1f}%"])
    s.append(table(rows, [3 * cm, 6.5 * cm, 6.5 * cm], st))
    s.append(Spacer(1, 4))
    s.append(P(
        "All agents were trained on one fixed district (statuses re-sampled each episode). On <b>unseen episodes of that "
        "district</b> the rankings hold. On <b>50 completely new district layouts</b> (new positions, populations and risks), "
        f"DQN drops sharply ({met['dqn']['newmap_return']:.1f}): its Q-values memorised the geometry of the training map. "
        f"<b>PPO generalises best</b> ({met['ppo']['newmap_return']:.1f}, {met['ppo']['newmap_served']:.1f}% served): its "
        "stochastic, entropy-regularised policy learned more transferable rules (go to broken, populous points; refuel "
        "before you run dry). The heuristic, which encodes those rules explicitly, is unaffected. Training on randomised maps "
        "would be the next step towards real deployment.", st["body"]))

    # ------------------------------------------------------------ 6 conclusion
    s.append(P("6. Conclusion and Discussion", st["h1"]))
    s.append(P(
        f"On this mission DQN performed best on the district it trained on: {f1(comp['DQN']['served_pct'])}% of "
        "people-days served, zero failed episodes, and ~42 km driven per repair versus 61 for the heuristic. It was also the most sample-efficient and the "
        "least sensitive to hyperparameters. This suits the problem: a small discrete action space, a dense population-weighted "
        "reward, and cheap simulation that a replay buffer can exploit. PPO was the strongest policy-gradient method and the "
        "best at generalising to new layouts, so it would be the better choice if one model must serve many districts. "
        "REINFORCE and A2C were limited by gradient variance. A2C also exposed a reward-specification loophole, a reminder "
        "that agents optimise the reward as written, not as intended.", st["body"]))
    s.append(P("<b>Improvements with more time:</b>", st["body"]))
    s.extend(bullets([
        "Charge <i>expected</i> future failures on early termination, which closes the stranding loophole, or add action "
        "masking for infeasible trips.",
        "Train on randomised districts and feed real risk predictions from the capstone classifier through the "
        "<font face='Courier'>points_csv</font> option.",
        "Use larger budgets and multiple seeds per configuration for confidence intervals, and try recurrent policies "
        "(LSTM) for the partial observability.",
        "Model multiple crews (multi-agent) and seasonal demand.",
    ], st))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.9 * cm, title="RL Summative Report - Water Point Dispatch", author=STUDENT)
    from pdf_style import page_decor

    deco = page_decor(f"{STUDENT} - Reinforcement Learning Summative: Water Point Maintenance Dispatch")
    doc.build(s, onFirstPage=deco, onLaterPages=deco)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
