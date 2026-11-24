import numpy as np
from gymnasium.utils.env_checker import check_env

from environment import WaterDispatchEnv, heuristic_action
from environment.custom_env import FUNCTIONAL, NEEDS_REPAIR, NON_FUNCTIONAL


def test_gymnasium_api():
    check_env(WaterDispatchEnv(), skip_render_check=True)


def test_spaces():
    env = WaterDispatchEnv()
    obs, _ = env.reset(seed=0)
    assert env.action_space.n == env.n + 5
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_horizon_truncation():
    env = WaterDispatchEnv()
    env.reset(seed=0)
    for day in range(env.cfg.horizon_days):
        _, _, term, trunc, _ = env.step(env.A_WAIT)
        assert not term
    assert trunc


def test_stranded_terminates_with_penalty():
    env = WaterDispatchEnv()
    env.reset(seed=0)
    env.fuel = 0.1
    _, r, term, _, info = env.step(0)
    assert term and info["stranded"] and r < -env.cfg.stranded_penalty


def test_major_repair_restores_broken_point():
    env = WaterDispatchEnv()
    env.reset(seed=0)
    env.step(0)
    env.status[0] = NON_FUNCTIONAL
    _, r, _, _, _ = env.step(env.A_MAJOR)
    assert env.status[0] == FUNCTIONAL and env.known[0] == FUNCTIONAL and r > 0


def test_minor_repair_cannot_fix_broken_point():
    env = WaterDispatchEnv()
    env.reset(seed=0)
    env.step(0)
    env.status[0] = NON_FUNCTIONAL
    env.step(env.A_MINOR)
    assert env.status[0] == NON_FUNCTIONAL and env.known[0] == NON_FUNCTIONAL


def test_inspect_reveals_status():
    env = WaterDispatchEnv()
    env.reset(seed=0)
    env.step(1)
    env.status[1] = NEEDS_REPAIR
    env.known[1] = FUNCTIONAL
    env.step(env.A_INSPECT)
    assert env.known[1] == env.status[1] or env.status[1] == NON_FUNCTIONAL


def test_invalid_actions_penalised():
    env = WaterDispatchEnv()
    for a in (env.A_INSPECT, env.A_MINOR, env.A_MAJOR, env.A_DEPOT):  # all invalid at the depot
        env.reset(seed=0)
        _, r_wait, _, _, _ = env.step(env.A_WAIT)
        env.reset(seed=0)
        _, r_bad, _, _, _ = env.step(a)
        assert np.isclose(r_bad, r_wait - env.cfg.invalid_penalty)


def test_heuristic_beats_waiting():
    def run(policy):
        env = WaterDispatchEnv()
        total = 0.0
        for s in range(10):
            env.reset(seed=s)
            done = False
            while not done:
                _, r, te, tr, _ = env.step(policy(env))
                total += r
                done = te or tr
        return total

    assert run(heuristic_action) > run(lambda e: e.A_WAIT)


def test_torchrl_specs():
    from torchrl.envs.utils import check_env_specs

    from training.common import make_env

    check_env_specs(make_env(0))
