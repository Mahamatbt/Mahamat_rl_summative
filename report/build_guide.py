"""Build the beginner's step-by-step guide PDF.

    uv run --with reportlab python report/build_guide.py
"""

from __future__ import annotations

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, SimpleDocTemplate, Spacer

from pdf_style import (
    ACCENT,
    ASSETS,
    FONT,
    FONT_B,
    GITHUB,
    INK,
    MUTED,
    ROOT,
    STUDENT,
    WARN_BG,
    P,
    bullets,
    callout,
    figure,
    page_decor,
    read_comparison,
    read_metrics,
    styles,
    table,
)

OUT = ROOT / "report" / "RL_Beginners_Guide.pdf"
st = styles(10.5)
W = A4[0] - 4.0 * cm


def code(text):
    return P(text.replace("\n", "<br/>").replace("  ", "&nbsp;&nbsp;"), st["code"])


def box(d, x, y, w, h, title, sub, fill):
    d.add(Rect(x, y, w, h, rx=8, ry=8, fillColor=colors.HexColor(fill), strokeColor=colors.HexColor("#1d2430"), strokeWidth=1))
    d.add(String(x + w / 2, y + h / 2 + 4, title, fontName=FONT_B, fontSize=12, textAnchor="middle", fillColor=INK))
    d.add(String(x + w / 2, y + h / 2 - 11, sub, fontName=FONT, fontSize=8.5, textAnchor="middle", fillColor=MUTED))


def arrow(d, x1, y1, x2, y2, label, label_dy=6):
    d.add(Line(x1, y1, x2, y2, strokeColor=ACCENT, strokeWidth=2))
    import math

    a = math.atan2(y2 - y1, x2 - x1)
    s = 7
    d.add(Polygon([x2, y2, x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4), x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4)],
                  fillColor=ACCENT, strokeColor=ACCENT))
    d.add(String((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label, fontName=FONT, fontSize=8.5, textAnchor="middle", fillColor=INK))


def loop_diagram():
    d = Drawing(W, 150)
    box(d, 0, 45, 125, 60, "AGENT", "the crew's brain", "#dcebff")
    box(d, W - 125, 45, 125, 60, "ENVIRONMENT", "district, pumps, truck", "#e3f5e8")
    arrow(d, 128, 90, W - 128, 90, "ACTION  (e.g. drive to WP09, major repair)")
    arrow(d, W - 128, 60, 128, 60, "STATE (what it sees) + REWARD (score)", -14)
    d.add(String(W / 2, 128, "One loop = one working day. 90 loops = one episode.", fontName=FONT_B, fontSize=9.5, textAnchor="middle", fillColor=ACCENT))
    return d


def algo_diagram():
    d = Drawing(W, 120)
    items = [
        ("DQN", "scores every move", "learns from a memory", "of past days", "#dcebff"),
        ("REINFORCE", "replays whole seasons", "does more of what", "worked overall", "#fde3d6"),
        ("A2C", "player + coach", "coach judges each", "move right away", "#fff0c9"),
        ("PPO", "careful coach", "improves in small,", "safe steps", "#d8f3e6"),
    ]
    w = (W - 30) / 4
    for i, (t, a, b, c, fill) in enumerate(items):
        x = i * (w + 10)
        d.add(Rect(x, 10, w, 100, rx=8, ry=8, fillColor=colors.HexColor(fill), strokeColor=colors.HexColor("#1d2430"), strokeWidth=0.8))
        d.add(String(x + w / 2, 85, t, fontName=FONT_B, fontSize=12, textAnchor="middle", fillColor=INK))
        for j, line in enumerate((a, b, c)):
            d.add(String(x + w / 2, 62 - 15 * j, line, fontName=FONT, fontSize=8.5, textAnchor="middle", fillColor=INK))
    return d


def build():
    comp = read_comparison()
    met = read_metrics()
    s = []

    # ------------------------------------------------------------ cover
    s.append(Spacer(1, 2.2 * cm))
    s.append(P("Teaching a Computer to Keep the Water Flowing", st["title"]))
    s.append(P("A step-by-step guide to this reinforcement learning project, written for someone who has never studied "
               "reinforcement learning.", st["subtitle"]))
    s.append(Spacer(1, 6))
    s.append(P(f"{STUDENT} - Reinforcement Learning Summative", st["subtitle"]))
    s.append(P(f'Code: <link href="{GITHUB}" color="#2a78d6">{GITHUB}</link>', st["subtitle"]))
    s.append(Spacer(1, 12))
    s.append(figure(ASSETS / "env_screenshot.png", 16.5, "The simulation built for this project: a maintenance truck looking after 10 rural water points.", st))
    s.append(Spacer(1, 8))
    s.append(P("<b>How to read this guide</b>", st["h2"]))
    s.extend(bullets([
        "Parts 1-3 explain the <b>problem</b> and the <b>basic ideas</b> of reinforcement learning, with no maths.",
        "Parts 4-6 explain the <b>simulation</b> we built and <b>how the computer learns</b> in it.",
        "Parts 7-9 explain the <b>four learning methods</b>, the <b>settings we tuned</b> and the <b>results</b>.",
        "Parts 10-12 are practical: how to <b>run the project</b>, how to <b>read the 3D view</b>, and a <b>script for the video</b>.",
        "A glossary of every technical word is at the end.",
    ], st))
    s.append(PageBreak())

    # ------------------------------------------------------------ 1 problem
    s.append(P("Part 1 - The real-world problem", st["h1"]))
    s.append(P(
        "In rural Tanzania, millions of people get their water from community water points: hand pumps, gravity-fed "
        "taps and motorised pumps. The national <i>Pump it Up</i> survey of about 59,000 water points found that only "
        "<b>54%</b> were working, <b>7%</b> were working but needed repair, and <b>38% were broken</b>.", st["body"]))
    s.append(P(
        "My capstone project built a <b>classifier</b>: a model that looks at a water point's details (how old it is, how "
        "it extracts water, whether the community pays for it) and predicts whether it is working. A prediction alone "
        "does not fix anything. A district usually has <b>one maintenance crew with one truck</b>, limited fuel, a limited "
        "budget and a few spare parts, and every day the crew has to decide:", st["body"]))
    s.append(callout("<b>\"Where should we go today, and what should we do when we get there?\"</b><br/>"
                     "Drive to the big village whose pump might be broken? Check on the old pump that hasn't been visited in a "
                     "month? Go back to the depot for fuel and parts first?", st))
    s.append(P(
        "These choices have long-term consequences. A trip uses fuel, so you may not reach the next village. Using your "
        "last major repair kit on a small village means you cannot fix a big one tomorrow. This kind of problem, a "
        "<b>sequence of decisions where today's choice changes tomorrow's options</b>, is exactly what reinforcement learning "
        "is designed for.", st["body"]))

    # ------------------------------------------------------------ 2 what is RL
    s.append(P("Part 2 - What is reinforcement learning?", st["h1"]))
    s.append(P(
        "<b>Reinforcement learning (RL)</b> is a way of teaching a computer by <b>trial and error with a score</b>, instead "
        "of by giving it the right answers. Think of how you might learn a new video game without reading the manual: you "
        "press buttons, watch what happens, notice when your score goes up or down, and over many games you work out a "
        "strategy. RL does the same thing, but the computer plays thousands of games.", st["body"]))
    s.append(P("Every RL problem has the same five ingredients:", st["body"]))
    s.append(table([
        ["Ingredient", "Plain meaning", "In our project"],
        ["<b>Agent</b>", "The decision-maker that learns", "The maintenance crew's \"brain\" (a neural network)"],
        ["<b>Environment</b>", "The world the agent acts in", "A simulated district: 10 water points, a depot, a truck"],
        ["<b>State / observation</b>", "What the agent can see right now", "Fuel, money, spare parts, what it last knew about each pump"],
        ["<b>Action</b>", "A choice the agent can make", "Drive somewhere, inspect, repair, refuel, or wait"],
        ["<b>Reward</b>", "A score for what just happened", "+ for fixing pumps, − for every person left without water"],
    ], [3.2 * cm, 5.5 * cm, W - 8.7 * cm], st))
    s.append(Spacer(1, 6))
    s.append(loop_diagram())
    s.append(P("The agent-environment loop: the agent acts, the world changes, and the agent gets a new view plus a score.", st["caption"]))
    s.append(P(
        "The agent's goal is not to get a high score <i>today</i>. It wants the highest <b>total</b> score over the whole "
        "<b>episode</b> (here, 90 days). That is what makes RL different from simple rules. A good agent will accept a "
        "small penalty now (a long, fuel-costly drive) to earn a big reward later (restoring water to 2,000 people).",
        st["body"]))
    s.append(callout(
        "<b>How is this different from my capstone classifier?</b> The classifier learns from a table of examples with known "
        "answers (\"this pump was broken\"). That is <i>supervised learning</i>. In RL nobody tells the agent the right action. "
        "It only sees the consequences of its own choices and has to work out which choices lead to good outcomes.", st))

    # ------------------------------------------------------------ 3 key ideas
    s.append(P("Part 3 - Five ideas you need", st["h1"]))
    s.append(P("1. Policy - the agent's strategy", st["h2"]))
    s.append(P(
        "A <b>policy</b> is a rule that says \"in this situation, do this\". At the start, the policy is random. Learning "
        "means gradually changing the policy so it picks better actions. Our policy is a <b>neural network</b>: a "
        "mathematical function with thousands of adjustable numbers (weights). It takes in the 86 numbers describing the "
        "situation and outputs a preference for each of the 15 actions.", st["body"]))
    s.append(P("2. Exploration vs exploitation", st["h2"]))
    s.append(P(
        "If the agent always does what it currently thinks is best (<b>exploitation</b>), it may never discover something "
        "better. If it always tries random things (<b>exploration</b>), it never uses what it learned. Imagine always eating "
        "at your favourite restaurant versus sometimes trying a new one. Every algorithm here balances the two, usually by "
        "exploring a lot at first and less later.", st["body"]))
    s.append(P("3. Discount factor (gamma, γ) - how much the future matters", st["h2"]))
    s.append(P(
        "A reward tomorrow is worth slightly less than a reward today. γ = 0.99 means the agent cares about rewards far "
        "ahead. γ = 0.95 means it focuses on roughly the next 20 days. This turned out to matter a lot in our experiments.",
        st["body"]))
    s.append(P("4. Value - how good is this situation?", st["h2"]))
    s.append(P(
        "A <b>value</b> is the agent's estimate of the total future score from a situation. \"Truck at a broken 2,000-person "
        "pump with a major repair kit\" has a high value. \"Truck far away with 2 litres of fuel\" has a very low value. Some "
        "algorithms learn values and act on them; others learn the policy directly.", st["body"]))
    s.append(P("5. Training vs testing", st["h2"]))
    s.append(P(
        "During <b>training</b> the agent plays hundreds of thousands of simulated days and adjusts its network. During "
        "<b>testing</b> we freeze it and measure it on <i>new</i> situations it has never seen. Only the test scores count, "
        "just as a student is judged on the exam, not the practice questions.", st["body"]))

    # ------------------------------------------------------------ 4 environment
    s.append(P("Part 4 - The simulation we built, step by step", st["h1"]))
    s.append(P(
        "Because we cannot let a beginner AI practise on real villages, we built a <b>simulator</b> (called the "
        "<i>environment</i>) in Python using the Gymnasium library. It works like this:", st["body"]))
    s.append(P("Step 1 - The district", st["h2"]))
    s.append(P(
        "A 30 km × 30 km district with a <b>depot</b> in the centre and <b>10 water points</b> around it. Each point serves "
        "between 100 and 4,000 people and has features taken from the Pump it Up data: pump age, extraction type (gravity, "
        "hand pump, motor pump...) and payment type (never pay, pay per bucket, monthly...). These are combined into a "
        "<b>failure-risk score</b> between 0 and 1. Old motor pumps in communities that never pay are the riskiest.",
        st["body"]))
    s.append(P("Step 2 - How an episode starts", st["h2"]))
    s.append(P(
        "The truck starts at the depot with a full tank (60 litres), 4 minor repair kits, 2 major repair kits and a $6,000 "
        "budget. Each pump is randomly made working, needs-repair or broken, using the same proportions as the real survey "
        "(54% / 7% / 38%). The crew knows these statuses from a survey that is 0-30 days old.", st["body"]))
    s.append(P("Step 3 - What happens each day", st["h2"]))
    s.append(P("The agent picks one of 15 actions:", st["body"]))
    s.append(table([
        ["Action", "What the crew does", "Cost / rule"],
        ["Drive to water point 0-9", "Travel to that village", "0.25 L fuel per km; a small penalty per km"],
        ["Return to depot", "Refuel and restock spare parts", "Paid from the budget ($1.50/L fuel, $40 minor kit, $250 major kit)"],
        ["Inspect", "Check the pump and learn its true status", "Small bonus if the pump is risky and hasn't been checked for a week"],
        ["Minor repair", "Fix a pump that \"needs repair\"", "1 minor kit + $20. Does NOT fix a fully broken pump"],
        ["Major repair", "Fix any broken pump", "1 major kit + $80. Wasteful on a pump with a small fault"],
        ["Wait", "Do nothing today", "-"],
    ], [3.6 * cm, 5.4 * cm, W - 9.0 * cm], st))
    s.append(Spacer(1, 4))
    s.append(P(
        "After the action, <b>the world moves on one day</b>. Every pump has a small chance of degrading (working → needs "
        "repair → broken), and riskier pumps degrade more often. <b>The crew does not see this happen.</b> It only knows a pump's "
        "status when it last inspected or repaired it, so information goes stale, just as in real life.", st["body"]))
    s.append(P("Step 4 - The score (reward)", st["h2"]))
    s.append(P("Each day the agent receives a reward that adds up these parts:", st["body"]))
    s.append(table([
        ["Event", "Reward", "Why"],
        ["Fix a broken pump", "+10 per 1,000 people it serves", "Restoring water is the mission"],
        ["Fix a needs-repair pump", "+5 per 1,000 people", "Prevents a full breakdown"],
        ["Every day, for each broken pump", "−0.5 per 1,000 people without water", "The main pressure: people waiting for water"],
        ["Driving", "−0.02 per km", "Fuel and time are not free"],
        ["Wasted repair or impossible action", "−1 to −1.5", "Discourages silly choices (e.g. repairing at the depot)"],
        ["Checking a risky, long-unseen pump", "up to +0.5", "Encourages gathering information"],
        ["Running out of fuel on the road", "−30 + the remaining days of downtime", "Must plan refuelling; quitting early can never pay"],
    ], [5.0 * cm, 5.3 * cm, W - 10.3 * cm], st))
    s.append(Spacer(1, 4))
    s.append(callout(
        "<b>Worked example (from a real run).</b> Day 1: the truck drives 10.5 km to WP08 (−0.21 for driving), while the "
        "four broken pumps cost −1.85 for the day, so the day scores <b>−2.06</b>. Day 2: a major repair at WP08, which serves "
        "1,156 people, earns +11.56. The pumps still broken cost −1.28, so the day scores <b>+10.28</b>. Over 90 days these add up "
        "to the <b>episode return</b>, the number we compare agents by.", st))
    s.append(P("Step 5 - How an episode ends", st["h2"]))
    s.extend(bullets([
        "Normally after <b>90 days</b>.",
        "Early, if the truck <b>runs out of fuel</b> away from the depot (stranded), or the <b>budget runs out</b>.",
    ], st))
    s.append(P("Step 6 - What the agent sees (the 86 numbers)", st["h2"]))
    s.append(P(
        "The neural network sees 86 numbers, all scaled between 0 and 1: where the truck is (11 numbers, one per location), "
        "fuel, budget, minor kits, major kits, the day, and for <b>each</b> of the 10 pumps: its last known status "
        "(3 numbers), how many days since it was checked, its population, its risk score and how far away it is. Notice "
        "what is <b>missing</b>: the true current status of each pump. The agent has to reason under uncertainty.",
        st["body"]))

    # ------------------------------------------------------------ 5 how it learns
    s.append(P("Part 5 - How the computer actually learns", st["h1"]))
    s.extend(bullets([
        "<b>1. Play.</b> The agent drives around the simulated district for some days, choosing actions with its current "
        "network plus some randomness (exploration). Every day is recorded as <i>(what it saw, what it did, the reward, "
        "what it saw next)</i>.",
        "<b>2. Judge.</b> The algorithm looks at those records and asks: which actions led to better totals than expected, "
        "and which led to worse?",
        "<b>3. Adjust.</b> It nudges the network's weights a little, so good actions become more likely (or are valued "
        "higher) in similar situations. The size of each nudge is the <b>learning rate</b>.",
        "<b>4. Repeat</b> about 300,000 simulated days (≈ 3,300 episodes of 90 days). On a laptop this takes 2-6 minutes per run, "
        "because the simulator is fast.",
        "<b>5. Check.</b> Every 10,000 days we pause and test the agent on 20 fixed practice episodes with no randomness, "
        "and keep the best version.",
    ], st))
    s.append(P(
        "We used <b>TorchRL</b>, a reinforcement learning library built on PyTorch. It provides the building blocks (data "
        "collectors, replay memory, loss functions for each algorithm), so all four methods share exactly the same simulator "
        "and the comparison is fair.", st["body"]))

    # ------------------------------------------------------------ 6 algorithms
    s.append(P("Part 6 - The four learning methods, in plain words", st["h1"]))
    s.append(algo_diagram())
    s.append(Spacer(1, 4))
    s.append(P("DQN - Deep Q-Network (value-based)", st["h2"]))
    s.append(P(
        "DQN learns a <b>scorecard</b>. For any situation, it estimates the long-term score of each of the 15 actions "
        "(the \"Q-values\"), then simply picks the action with the highest score. Three tricks make it work:", st["body"]))
    s.extend(bullets([
        "<b>Replay memory:</b> it stores the last 50,000 days and repeatedly studies random samples from them, like "
        "revising old exam papers. Each experience gets reused many times, so DQN learns from relatively little play.",
        "<b>Target network:</b> it compares its guesses against a slowly-updated copy of itself, which stops it from "
        "\"chasing its own tail\".",
        "<b>ε-greedy exploration:</b> it starts by acting 100% randomly and slowly reduces that to 5%.",
    ], st))
    s.append(P("REINFORCE (policy gradient)", st["h2"]))
    s.append(P(
        "REINFORCE skips the scorecard and adjusts the policy directly. It plays complete 90-day episodes, adds up the "
        "score, and makes every action from a good episode a bit more likely and every action from a bad one a bit less "
        "likely. It is simple, but noisy: a great day might be credited to a bad decision just because the whole episode "
        "went well. That noise (\"high variance\") makes it the slowest learner here.", st["body"]))
    s.append(P("A2C - Advantage Actor-Critic", st["h2"]))
    s.append(P(
        "A2C uses two networks: an <b>actor</b> (the player, which picks actions) and a <b>critic</b> (the coach, which "
        "estimates how good each situation is). After a short burst of play (64 days), the coach says \"that move turned "
        "out better or worse than I expected\" (the <b>advantage</b>), and the actor adjusts. This gives faster, less noisy "
        "feedback than REINFORCE.", st["body"]))
    s.append(P("PPO - Proximal Policy Optimisation", st["h2"]))
    s.append(P(
        "PPO is actor-critic with a <b>safety rule</b>: each update may only change the policy by a small amount (the "
        "\"clip\", 20%). It collects about 2,000 days of play, then studies that batch 10 times in small pieces. The safety "
        "rule stops it from wrecking a good strategy with one over-confident update. It is the most widely used method in "
        "modern RL.", st["body"]))

    # ------------------------------------------------------------ 7 hyperparameters
    s.append(P("Part 7 - Tuning the settings (hyperparameters)", st["h1"]))
    s.append(P(
        "<b>Hyperparameters</b> are settings that we choose before training, like the oven temperature and baking time in a "
        "recipe. The assignment required at least 10 different combinations per algorithm. We ran exactly 10 for each "
        "(40 training runs, about 2 hours of computing). The most important settings:", st["body"]))
    s.append(table([
        ["Setting", "What it controls", "What we saw"],
        ["Learning rate", "How big each adjustment is", "Too big (DQN 0.001) = unstable; too small (PPO, A2C 0.0001) = barely learns"],
        ["Gamma (γ)", "How far ahead the agent plans", "0.95 was best for REINFORCE and A2C: shorter planning gave less noisy learning"],
        ["Batch / replay size", "How much memory DQN studies from", "A larger memory (100k) learned more slowly"],
        ["Exploration (ε)", "How long DQN acts randomly", "Reducing randomness sooner (15% of training) helped"],
        ["Entropy bonus", "Reward for keeping options open", "Too high (0.05) kept PPO too random"],
        ["Clip (PPO)", "Maximum change per update", "0.1 too cautious; 0.3 better"],
        ["GAE λ", "How the coach blends short and long-term feedback", "0.8 gave PPO its best run"],
        ["n-step (A2C)", "Days of play before each update", "Too short (16) led to the \"strand the truck\" trick (Part 8)"],
    ], [3.2 * cm, 5.0 * cm, W - 8.2 * cm], st))
    s.append(Spacer(1, 4))
    s.append(figure(ASSETS / "sweep_summary.png", 15.5, "Best test score of each of the 10 settings per algorithm (further right = better).", st))

    # ------------------------------------------------------------ 8 results
    s.append(P("Part 8 - What happened: the results", st["h1"]))
    s.append(P(
        "We tested the best version of each algorithm on <b>100 brand-new episodes</b> and compared it with two simple "
        "baselines: a <b>random</b> agent and a <b>hand-written rule</b> (\"go to the biggest broken village, check old pumps, "
        "refuel when low\"), which represents what a sensible human planner might do.", st["body"]))
    rows = [["Agent", "Average score", "People-days with water", "Repairs per 90 days"]]
    for a in ["DQN", "PPO", "Heuristic", "A2C", "REINFORCE", "Random"]:
        r = comp[a]
        lab = "Hand-written rule" if a == "Heuristic" else a
        rows.append([f"<b>{lab}</b>", f"{float(r['mean_return']):.1f}", f"{float(r['served_pct']):.1f}%", f"{float(r['repairs']):.1f}"])
    s.append(table(rows, [4.5 * cm, 3.5 * cm, 4.5 * cm, 4.0 * cm], st, highlight_row=1))
    s.append(Spacer(1, 6))
    s.append(figure(ASSETS / "learning_curves.png", 14.5,
                    "How each algorithm's test score improved during training. The dashed line is the hand-written rule.", st))
    s.append(P("What this means in plain words:", st["body"]))
    s.extend(bullets([
        f"<b>DQN won.</b> It kept water flowing for {float(comp['DQN']['served_pct']):.1f}% of people-days and beat the "
        "hand-written rule. It made more repairs while driving about a quarter fewer kilometres, because it learned to "
        "chain nearby repairs instead of zig-zagging back to the depot.",
        "<b>PPO came second</b> and roughly matched the hand-written rule.",
        "<b>A2C and REINFORCE improved a lot</b> compared with random play, but were still learning when training stopped. "
        "They need more practice because they cannot re-use old experience the way DQN's replay memory does.",
        f"<b>Surprise 1 - generalisation.</b> When we tested on <i>completely new maps</i> (villages in different places), "
        f"DQN's score dropped to {met['dqn']['newmap_return']:.0f}, but PPO held up best ({met['ppo']['newmap_return']:.0f}). "
        "DQN had partly memorised the training map; PPO had learned more general habits.",
        "<b>Surprise 2 - reward hacking.</b> Some A2C settings discovered a trick: fix a few pumps, then deliberately run "
        "out of fuel so the episode ends early. The penalty for stopping only counted pumps broken <i>at that moment</i>, "
        "not the ones that would have broken later. This is a famous issue in RL: the agent optimises the score as "
        "<i>written</i>, not as <i>intended</i>. None of the best models use this trick.",
    ], st))

    # ------------------------------------------------------------ 9 3d view
    s.append(P("Part 9 - Reading the 3D simulation", st["h1"]))
    s.append(figure(ASSETS / "env_closeup.png", 14.5, "Close-up: a selected water point with its details panel.", st))
    s.append(table([
        ["What you see", "What it means"],
        ["Coloured band on each water tank", "The TRUE status: green = working, amber = needs repair, red = broken"],
        ["Flashing beacon on top of a tank", "That pump is not working (fast red = broken, slow amber = needs repair)"],
        ["Floating diamond above a tank", "What the crew BELIEVES; it fades to grey as the information gets old"],
        ["Huts around a tank", "The village; more huts = more people served"],
        ["Villagers with buckets at the tap", "Water is flowing; at a broken tap they stand waiting further away"],
        ["White truck + dust trail + blue line", "The crew, where it is driving, and its recent route"],
        ["Cyan rings / yellow-orange sparks", "An inspection / a repair is happening"],
        ["Left panel", "Day, fuel, budget, kits, last action and its reward, % of people with water, minimap"],
    ], [5.5 * cm, W - 5.5 * cm], st))
    s.append(Spacer(1, 6))
    s.append(P("<b>Controls:</b>", st["body"]))
    s.append(table([
        ["Control", "Effect", "Control", "Effect"],
        ["Left-drag", "Turn around the scene (orbit)", "Click a tank", "Show its details panel"],
        ["Right-drag or W A S D", "Move the camera (pan)", "TAB", "Select the next water point"],
        ["Scroll or Z / X", "Zoom in / out", "F", "Follow the truck"],
        ["Q / E", "Rotate", "SPACE", "Pause / resume"],
        ["T / G", "Tilt up / down", "+ / -", "Faster / slower animation"],
        ["R", "Reset the camera", "H / ESC", "Hide help / quit"],
    ], [3.6 * cm, 4.9 * cm, 3.0 * cm, W - 11.5 * cm], st))

    # ------------------------------------------------------------ 10 run it
    s.append(P("Part 10 - Running the project yourself, step by step", st["h1"]))
    s.append(P("<b>Step 1 - Install uv</b> (a tool that installs Python and all libraries automatically):", st["body"]))
    s.append(code("curl -LsSf https://astral.sh/uv/install.sh | sh"))
    s.append(P("<b>Step 2 - Download the project and install everything:</b>", st["body"]))
    s.append(code(f"git clone {GITHUB}.git\ncd Mahamat_rl_summative\nuv sync"))
    s.append(P("<b>Step 3 - Watch the best agent</b> (opens the 3D window; the terminal prints every day's decision):", st["body"]))
    s.append(code("uv run main.py"))
    s.append(P("<b>Other useful commands:</b>", st["body"]))
    s.append(code("uv run main.py --algo ppo          # watch PPO (also: dqn, a2c, reinforce)\n"
                  "uv run main.py --episodes 1        # just one 90-day episode\n"
                  "uv run main.py --no-render         # terminal only, no 3D window\n"
                  "uv run main.py compare             # re-test all agents, redraw the charts\n"
                  "uv run main.py train --algo all    # retrain all 40 runs (about 2 hours)\n"
                  "uv run pytest                      # run the automatic checks"))
    s.append(P("<b>Where things are:</b>", st["body"]))
    s.append(table([
        ["Folder / file", "Contents"],
        ["environment/custom_env.py", "The simulator: rules, actions, rewards"],
        ["environment/rendering.py", "The 3D visualisation"],
        ["environment/water_points.py", "Water point generation and the risk score"],
        ["training/dqn_training.py", "DQN training + its 10 settings"],
        ["training/pg_training.py", "REINFORCE, PPO and A2C training + their 10 settings each"],
        ["models/", "The trained agents (best_dqn.pt, best_ppo.pt, ...)"],
        ["logs/", "Score history of every training run (CSV files)"],
        ["assets/", "Screenshots and charts"],
        ["main.py", "The starting point that ties everything together"],
    ], [5.5 * cm, W - 5.5 * cm], st))

    # ------------------------------------------------------------ 11 video script
    s.append(P("Part 11 - Script for the 5-minute video", st["h1"]))
    s.append(callout(
        "<b>Before recording:</b> share your <b>entire screen</b> with the <b>camera on</b>. Have a terminal open in the project "
        "folder and close unrelated windows. The assignment requires that running the agent takes up about 3/4 of the video "
        "(around 3 minutes of 5), and that the viewer can see you navigate the folders and run the Python file.", st, bg=WARN_BG,
        border=colors.HexColor("#eda100")))
    s.append(table([
        ["Time", "Show", "Say (in your own words)"],
        ["0:00-0:30", "Your face + the README", "\"The problem: rural water points break, and a single crew with limited fuel, budget and parts must "
                                                 "decide where to go each day. This extends my capstone, which predicts which pumps fail.\""],
        ["0:30-1:00", "custom_env.py (scroll)", "\"The agent is the crew. It has 15 actions: drive to a point, return to the depot, inspect, minor "
                                                "or major repair, wait. Its objective is to maximise people-days with water over 90 days.\""],
        ["1:00-1:15", "Reward section of the README", "\"Reward: plus for restoring water, scaled by population; minus every day for people without "
                                                      "water; small costs for driving and wasted actions; a big penalty for running out of fuel.\""],
        ["1:15-4:15", "Terminal: <font face='Courier'>ls</font>, <font face='Courier'>cd</font>, then "
                      "<font face='Courier'>uv run main.py</font> (3 episodes, about 1 minute each); the 3D window + terminal side by side",
         "Narrate while it runs, for example: \"It's going to WP09 first: 2,135 people and a high risk score... now a major repair... it returns to "
         "refuel before the tank runs low...\" Orbit the camera, click a tank to show the details, point out the diamond (belief) vs "
         "the tank colour (truth)."],
        ["4:15-5:00", "The end-of-episode summary + learning_curves.png",
         f"\"DQN was best: {float(comp['DQN']['served_pct']):.0f}% of people-days served, beating a hand-written rule. "
         "PPO generalised best to new maps. Some A2C settings learned to reward-hack by stranding the truck.\""],
    ], [2.0 * cm, 4.5 * cm, W - 6.5 * cm], st))

    # ------------------------------------------------------------ 12 glossary
    s.append(P("Part 12 - Glossary", st["h1"]))
    gl = [
        ("Action", "A choice the agent can make in one step."),
        ("Advantage", "How much better an action turned out than the agent expected."),
        ("Agent", "The learner and decision-maker."),
        ("Baseline (in REINFORCE)", "An estimate of the average score, subtracted to reduce noise."),
        ("Critic", "A network that estimates how good a situation is (its value)."),
        ("Discount factor (γ)", "How much future rewards count compared with immediate ones."),
        ("Entropy", "How random or undecided the policy is. High = exploring, low = confident."),
        ("Environment", "The simulated world the agent interacts with."),
        ("Episode", "One complete run from start to end (here up to 90 days)."),
        ("Epsilon (ε)", "In DQN, the probability of taking a random action to explore."),
        ("Generalisation", "Performing well on situations not seen during training."),
        ("Gymnasium", "A standard Python library for defining RL environments."),
        ("Hyperparameter", "A setting chosen before training (learning rate, γ, batch size...)."),
        ("Learning rate", "How big each adjustment to the network is."),
        ("Neural network", "A function with many adjustable weights that maps observations to decisions."),
        ("Observation / state", "The numbers describing what the agent can see."),
        ("Off-policy / on-policy", "Whether the agent can learn from old experience (DQN) or only from its current behaviour (PPO, A2C, REINFORCE)."),
        ("Policy", "The agent's strategy: what to do in each situation."),
        ("Q-value", "DQN's estimate of the total future score of taking an action in a situation."),
        ("Replay buffer", "DQN's memory of past experiences, reused for learning."),
        ("Return", "The total reward collected over an episode."),
        ("Reward", "The score the environment gives after each action."),
        ("Reward hacking", "When an agent finds an unintended shortcut to a high score."),
        ("TorchRL", "The PyTorch library used to implement all four algorithms."),
    ]
    s.append(table([["Term", "Meaning"]] + [[f"<b>{a}</b>", b] for a, b in gl], [4.6 * cm, W - 4.6 * cm], st))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=2 * cm,
                            title="Teaching a Computer to Keep the Water Flowing - Beginner's Guide", author=STUDENT)
    deco = page_decor("Beginner's guide - Water Point Maintenance Dispatch with Reinforcement Learning")
    doc.build(s, onFirstPage=deco, onLaterPages=deco)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
