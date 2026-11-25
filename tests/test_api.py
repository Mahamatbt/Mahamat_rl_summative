import pytest

from deployment.api import MODEL_PATHS, DispatchAgent
from environment import WaterDispatchEnv


@pytest.mark.parametrize("algo", list(MODEL_PATHS))
def test_agent_matches_env(algo):
    env = WaterDispatchEnv()
    obs, _ = env.reset(seed=123)
    agent = DispatchAgent(algo)
    for _ in range(20):
        rec = agent.recommend(obs)
        assert 0 <= rec["action"] < env.action_space.n
        assert rec["action_name"] == env.action_name(rec["action"])
        assert len(rec["scores"]) == env.action_space.n
        obs, _, term, trunc, _ = env.step(rec["action"])
        if term or trunc:
            break


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        DispatchAgent("dqn").recommend([0.0] * 5)
