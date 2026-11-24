"""Water Point Maintenance Dispatch environment (Gymnasium).

A single maintenance crew (one pickup truck) looks after N rural water points
in a district. Each water point is functional, needs repair, or non
functional, and degrades stochastically each day according to its failure-risk
score. The crew only knows the status it last observed (survey or inspection),
so it has to balance travelling, inspecting and repairing under fuel, budget
and spare-part constraints to keep as many people as possible supplied with
water over a 90-day planning horizon.

One environment step = one working day.

Actions (Discrete(N + 5))
-------------------------
0 .. N-1  drive to water point k
N         return to depot (refuel the truck and restock spare-part kits)
N + 1     inspect the water point the crew is at (reveals true status)
N + 2     minor repair (1 minor kit) - fixes "needs repair" only
N + 3     major repair (1 major kit) - fixes "needs repair" and "non functional"
N + 4     wait / stand by for the day

Observation (Box, float32, in [0, 1])
-------------------------------------
location one-hot (N + 1, last slot = depot)
fuel, budget, minor kits, major kits (fractions of capacity), day / horizon
for each water point:
    last known status one-hot (3), days since last observation / horizon,
    population / max population, failure-risk score, distance from crew / max distance
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.water_points import STATUS_PROBS, WaterPoint, generate_points, load_points_csv

FUNCTIONAL, NEEDS_REPAIR, NON_FUNCTIONAL = 0, 1, 2
STATUS_NAMES = ["functional", "needs repair", "non functional"]


@dataclass
class EnvConfig:
    n_points: int = 10
    map_size_km: float = 30.0
    horizon_days: int = 90
    map_seed: int = 7  # fixed district layout; statuses are resampled every episode
    randomize_map: bool = False
    points_csv: str | None = None

    # truck and logistics
    fuel_capacity_l: float = 60.0
    fuel_per_km_l: float = 0.25
    fuel_price: float = 1.5  # USD per litre, paid when refuelling at the depot
    minor_kit_capacity: int = 4
    major_kit_capacity: int = 2
    minor_kit_cost: float = 40.0
    major_kit_cost: float = 250.0
    minor_labour_cost: float = 20.0
    major_labour_cost: float = 80.0
    start_budget: float = 6000.0

    # degradation (per day)
    p_fail_base: float = 0.004
    p_fail_risk: float = 0.03  # functional -> needs repair = base + risk_coef * risk
    p_break_base: float = 0.03
    p_break_risk: float = 0.07  # needs repair -> non functional

    # reward weights (populations are expressed in thousands of people)
    restore_reward: float = 10.0  # per 1000 people restored from non functional
    downtime_penalty: float = 0.5  # per 1000 people without water per day
    degraded_weight: float = 0.3  # "needs repair" counts as partial downtime
    travel_penalty: float = 0.02  # per km driven
    invalid_penalty: float = 1.0
    wasted_repair_penalty: float = 1.5
    inspect_bonus: float = 0.5  # x risk, only when the point was unobserved for >= stale_days
    stale_days: int = 7
    stranded_penalty: float = 30.0
    bankrupt_penalty: float = 10.0

    survey_age_max_days: int = 30
    extra: dict = field(default_factory=dict)


class WaterDispatchEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None, **overrides):
        super().__init__()
        cfg = config or EnvConfig()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise TypeError(f"Unknown EnvConfig field: {k}")
            setattr(cfg, k, v)
        self.cfg = cfg
        self.render_mode = render_mode
        self._renderer = None

        if cfg.points_csv:
            self.points = load_points_csv(cfg.points_csv)
            cfg.n_points = len(self.points)
        else:
            self.points = generate_points(cfg.n_points, cfg.map_size_km, np.random.default_rng(cfg.map_seed))
        self.n = cfg.n_points
        self._set_geometry()

        # action indices
        self.A_DEPOT = self.n
        self.A_INSPECT = self.n + 1
        self.A_MINOR = self.n + 2
        self.A_MAJOR = self.n + 3
        self.A_WAIT = self.n + 4
        self.action_space = spaces.Discrete(self.n + 5)

        obs_dim = (self.n + 1) + 5 + 7 * self.n
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _set_geometry(self):
        cfg = self.cfg
        self.depot_xy = np.array([cfg.map_size_km / 2, cfg.map_size_km / 2])
        coords = [np.array([p.x_km, p.y_km]) for p in self.points] + [self.depot_xy]
        self.coords = np.stack(coords)  # index n = depot
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.dist = np.linalg.norm(diff, axis=-1)
        self.max_dist = float(self.dist.max())
        self.pop = np.array([p.population for p in self.points], dtype=np.float64)
        self.pop_k = self.pop / 1000.0
        self.risk = np.array([p.risk for p in self.points], dtype=np.float64)
        self.p_fail = cfg.p_fail_base + cfg.p_fail_risk * self.risk
        self.p_break = cfg.p_break_base + cfg.p_break_risk * self.risk

    def action_name(self, a: int) -> str:
        a = int(a)
        if a < self.n:
            return f"drive to WP{a:02d}"
        return {
            self.A_DEPOT: "return to depot",
            self.A_INSPECT: "inspect",
            self.A_MINOR: "minor repair",
            self.A_MAJOR: "major repair",
            self.A_WAIT: "wait",
        }[a]

    def location_name(self, loc: int | None = None) -> str:
        loc = self.loc if loc is None else loc
        return "depot" if loc == self.n else f"WP{loc:02d}"

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg = self.cfg
        if cfg.randomize_map and not cfg.points_csv:
            self.points = generate_points(cfg.n_points, cfg.map_size_km, self.np_random)
            self._set_geometry()

        self.status = self.np_random.choice(3, size=self.n, p=STATUS_PROBS)
        self.known = self.status.copy()  # last survey results
        self.days_since = self.np_random.integers(0, cfg.survey_age_max_days + 1, size=self.n).astype(np.float64)
        self.loc = self.n
        self.prev_loc = self.n
        self.day = 0
        self.fuel = cfg.fuel_capacity_l
        self.budget = cfg.start_budget
        self.minor_kits = cfg.minor_kit_capacity
        self.major_kits = cfg.major_kit_capacity

        self.last_action: int | None = None
        self.last_event = "crew at depot, ready"
        self.last_reward = 0.0
        self.episode_return = 0.0
        self.people_days_served = 0.0
        self.repairs_done = 0
        self.km_driven = 0.0
        self.stranded = False

        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def step(self, action):
        cfg = self.cfg
        a = int(action)
        assert self.action_space.contains(a), f"invalid action {action}"
        self.prev_loc = self.loc
        self.last_action = a
        self.days_since += 1
        reward = 0.0
        terminated = False
        repaired_today = -1
        event = ""

        if a < self.n or a == self.A_DEPOT:
            target = a
            if target == self.loc:
                reward -= cfg.invalid_penalty
                event = f"already at {self.location_name(target)} (wasted day)"
            else:
                d = self.dist[self.loc, target]
                fuel_needed = d * cfg.fuel_per_km_l
                reward -= cfg.travel_penalty * d
                if fuel_needed > self.fuel + 1e-9:
                    # truck runs dry on the road: crew stranded away from the depot
                    self.km_driven += self.fuel / cfg.fuel_per_km_l
                    self.fuel = 0.0
                    self.stranded = True
                    terminated = True
                    reward -= cfg.stranded_penalty
                    event = f"ran out of fuel on the way to {self.location_name(target)}!"
                else:
                    self.fuel -= fuel_needed
                    self.km_driven += d
                    self.loc = target
                    event = f"drove {d:.1f} km to {self.location_name(target)}"
                    if target == self.A_DEPOT:
                        event += "; " + self._restock()
        elif a == self.A_INSPECT:
            if self.loc == self.n:
                reward -= cfg.invalid_penalty
                event = "nothing to inspect at the depot"
            else:
                k = self.loc
                if self.days_since[k] - 1 >= cfg.stale_days:
                    reward += cfg.inspect_bonus * self.risk[k]
                self._observe(k)
                event = f"inspected WP{k:02d}: {STATUS_NAMES[self.status[k]]}"
        elif a in (self.A_MINOR, self.A_MAJOR):
            r, event, repaired_today = self._repair(major=(a == self.A_MAJOR))
            reward += r
        else:  # wait
            event = "standing by"

        # daily dynamics: water points degrade
        u = self.np_random.random(self.n)
        for k in range(self.n):
            if k == repaired_today:
                continue
            if self.status[k] == FUNCTIONAL and u[k] < self.p_fail[k]:
                self.status[k] = NEEDS_REPAIR
            elif self.status[k] == NEEDS_REPAIR and u[k] < self.p_break[k]:
                self.status[k] = NON_FUNCTIONAL

        down = np.where(self.status == NON_FUNCTIONAL, 1.0, np.where(self.status == NEEDS_REPAIR, cfg.degraded_weight, 0.0))
        reward -= cfg.downtime_penalty * float(np.sum(down * self.pop_k))
        self.people_days_served += float(np.sum(self.pop[self.status != NON_FUNCTIONAL]))

        if self.budget <= 0 and not terminated:
            terminated = True
            reward -= cfg.bankrupt_penalty
            event += "; repair budget exhausted"

        self.day += 1
        if terminated:
            # Ending early must never be cheaper than continuing: the communities keep
            # suffering the current outages for the rest of the horizon.
            remaining = cfg.horizon_days - self.day
            reward -= cfg.downtime_penalty * float(np.sum(down * self.pop_k)) * remaining
        truncated = self.day >= cfg.horizon_days and not terminated

        self.last_event = event
        self.last_reward = reward
        self.episode_return += reward
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------ mechanics
    def _observe(self, k: int):
        self.known[k] = self.status[k]
        self.days_since[k] = 0

    def _restock(self) -> str:
        cfg = self.cfg
        spent = 0.0
        litres = min(cfg.fuel_capacity_l - self.fuel, max(self.budget, 0) / cfg.fuel_price)
        self.fuel += litres
        spent += litres * cfg.fuel_price
        self.budget -= litres * cfg.fuel_price
        while self.minor_kits < cfg.minor_kit_capacity and self.budget >= cfg.minor_kit_cost:
            self.minor_kits += 1
            self.budget -= cfg.minor_kit_cost
            spent += cfg.minor_kit_cost
        while self.major_kits < cfg.major_kit_capacity and self.budget >= cfg.major_kit_cost:
            self.major_kits += 1
            self.budget -= cfg.major_kit_cost
            spent += cfg.major_kit_cost
        return f"refuelled and restocked (${spent:.0f})"

    def _repair(self, major: bool):
        cfg = self.cfg
        kind = "major" if major else "minor"
        if self.loc == self.n:
            return -cfg.invalid_penalty, f"cannot do a {kind} repair at the depot", -1
        kits = self.major_kits if major else self.minor_kits
        if kits <= 0:
            return -cfg.invalid_penalty, f"no {kind} repair kits left", -1
        labour = cfg.major_labour_cost if major else cfg.minor_labour_cost
        if self.budget < labour:
            return -cfg.invalid_penalty, "not enough budget for labour", -1

        k = self.loc
        if major:
            self.major_kits -= 1
        else:
            self.minor_kits -= 1
        self.budget -= labour
        before = self.status[k]
        reward = 0.0

        if before == FUNCTIONAL:
            reward -= cfg.wasted_repair_penalty
            msg = f"{kind} repair at WP{k:02d} wasted: point was functional"
        elif before == NEEDS_REPAIR:
            self.status[k] = FUNCTIONAL
            reward += 0.5 * cfg.restore_reward * self.pop_k[k]
            if major:
                reward -= 0.5 * cfg.wasted_repair_penalty  # overkill: a minor kit would do
            msg = f"{kind} repair at WP{k:02d}: restored ({self.points[k].population} people)"
            self.repairs_done += 1
        else:  # NON_FUNCTIONAL
            if major:
                self.status[k] = FUNCTIONAL
                reward += cfg.restore_reward * self.pop_k[k]
                msg = f"major repair at WP{k:02d}: restored ({self.points[k].population} people)"
                self.repairs_done += 1
            else:
                reward -= cfg.wasted_repair_penalty
                msg = f"minor repair at WP{k:02d} failed: pump needs a major repair"
        self._observe(k)
        return reward, msg, k

    # ------------------------------------------------------------------ obs/info
    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        loc = np.zeros(self.n + 1, dtype=np.float32)
        loc[self.loc] = 1.0
        scalars = np.array(
            [
                self.fuel / cfg.fuel_capacity_l,
                np.clip(self.budget / cfg.start_budget, 0, 1),
                self.minor_kits / cfg.minor_kit_capacity,
                self.major_kits / cfg.major_kit_capacity,
                self.day / cfg.horizon_days,
            ],
            dtype=np.float32,
        )
        known = np.eye(3, dtype=np.float32)[self.known]  # (n, 3)
        per_point = np.column_stack(
            [
                known,
                np.clip(self.days_since / cfg.horizon_days, 0, 1),
                self.pop / self.pop.max(),
                self.risk,
                self.dist[self.loc, : self.n] / self.max_dist,
            ]
        ).astype(np.float32)
        return np.clip(np.concatenate([loc, scalars, per_point.ravel()]), 0.0, 1.0)

    def _info(self) -> dict:
        functional_pop = float(np.sum(self.pop[self.status != NON_FUNCTIONAL]))
        return {
            "day": self.day,
            "location": self.location_name(),
            "fuel": self.fuel,
            "budget": self.budget,
            "minor_kits": self.minor_kits,
            "major_kits": self.major_kits,
            "n_functional": int(np.sum(self.status == FUNCTIONAL)),
            "n_needs_repair": int(np.sum(self.status == NEEDS_REPAIR)),
            "n_non_functional": int(np.sum(self.status == NON_FUNCTIONAL)),
            "pct_population_served": 100.0 * functional_pop / float(self.pop.sum()),
            "people_days_served": self.people_days_served,
            "repairs_done": self.repairs_done,
            "km_driven": self.km_driven,
            "stranded": self.stranded,
        }

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import WaterDispatchRenderer

            self._renderer = WaterDispatchRenderer(self, mode=self.render_mode)
        return self._renderer.draw()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def heuristic_action(env: WaterDispatchEnv) -> int:
    """Rule-based dispatcher used as a baseline (what a planner might do by hand).

    Go to the most-populated point believed to be broken (or long unobserved and
    risky), inspect if stale, repair with the right kit, return to depot when
    fuel or kits run low.
    """
    cfg = env.cfg
    at_point = env.loc < env.n
    if at_point:
        k = env.loc
        if env.days_since[k] > 0 and env.known[k] == FUNCTIONAL and env.days_since[k] >= cfg.stale_days:
            return env.A_INSPECT
        if env.known[k] == NON_FUNCTIONAL and env.major_kits > 0:
            return env.A_MAJOR
        if env.known[k] == NEEDS_REPAIR and env.minor_kits > 0:
            return env.A_MINOR
        if env.known[k] == NEEDS_REPAIR and env.major_kits > 0:
            return env.A_MAJOR
    # choose next target
    priority = env.pop_k * (
        np.where(env.known == NON_FUNCTIONAL, 1.0 if env.major_kits > 0 else 0.0, 0.0)
        + np.where(env.known == NEEDS_REPAIR, 0.6 if (env.minor_kits + env.major_kits) > 0 else 0.0, 0.0)
        + np.where(env.known == FUNCTIONAL, 0.02 * env.risk * env.days_since, 0.0)
    )
    if at_point:
        priority[env.loc] = -1
    k = int(np.argmax(priority))
    reserve = env.dist[k, env.n] * cfg.fuel_per_km_l
    need = env.dist[env.loc, k] * cfg.fuel_per_km_l
    if priority[k] <= 0.05 or env.fuel < need + reserve + 1.0 or (env.major_kits == 0 and env.minor_kits == 0):
        return env.A_DEPOT if env.loc != env.n else (k if priority[k] > 0.05 and env.fuel >= need + reserve else env.A_WAIT)
    return k
