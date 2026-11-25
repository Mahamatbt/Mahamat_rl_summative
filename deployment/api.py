"""Deployment API for a trained dispatch policy.

Serialization path
------------------
Training writes ``models/<dqn|pg>/best_<algo>.pt``: a ``torch.save`` dict holding
``state_dict``, the run ``config`` (hidden sizes, activation), ``obs_dim``,
``n_actions`` and eval ``metrics``. ``best_<algo>.json`` stores the same metrics
as plain JSON. ``DispatchAgent`` rebuilds the MLP from that dict, so no training
code or TorchRL objects are needed at inference time.

Python use
----------
    from deployment.api import DispatchAgent
    agent = DispatchAgent("dqn")
    rec = agent.recommend(obs)      # obs: 86 floats from WaterDispatchEnv
    rec["action"], rec["action_name"], rec["scores"]

HTTP use (for a dashboard / mobile front end), standard library only
--------------------------------------------------------------------
    uv run python -m deployment.api --algo dqn --port 8000
    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/recommend -d '{"observation": [... 86 floats ...]}'
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch

from training.common import load_policy_net

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATHS = {
    "dqn": ROOT / "models" / "dqn" / "best_dqn.pt",
    "reinforce": ROOT / "models" / "pg" / "best_reinforce.pt",
    "ppo": ROOT / "models" / "pg" / "best_ppo.pt",
    "a2c": ROOT / "models" / "pg" / "best_a2c.pt",
}
N_POINTS = 10


def action_name(a: int, n_points: int = N_POINTS) -> str:
    if a < n_points:
        return f"drive to WP{a:02d}"
    return ["return to depot", "inspect", "minor repair", "major repair", "wait"][a - n_points]


class DispatchAgent:
    """Greedy inference wrapper around a saved checkpoint."""

    def __init__(self, algo: str = "dqn", path: str | Path | None = None):
        self.algo = algo
        self.path = Path(path) if path else MODEL_PATHS[algo]
        self.net, ckpt = load_policy_net(self.path)
        self.obs_dim = ckpt["obs_dim"]
        self.n_actions = ckpt["n_actions"]
        self.run = ckpt["config"]["name"]
        self.metrics = ckpt["metrics"]

    def act(self, obs) -> int:
        return self.recommend(obs)["action"]

    def recommend(self, obs) -> dict:
        x = np.asarray(obs, dtype=np.float32)
        if x.shape != (self.obs_dim,):
            raise ValueError(f"observation must have shape ({self.obs_dim},), got {x.shape}")
        with torch.no_grad():
            out = self.net(torch.from_numpy(x).unsqueeze(0)).squeeze(0)
        a = int(out.argmax().item())
        # DQN outputs Q-values; PG actors output logits -> report probabilities
        scores = out if self.algo == "dqn" else torch.softmax(out, -1)
        return {
            "action": a,
            "action_name": action_name(a, self.n_actions - 5),
            "score_type": "q_value" if self.algo == "dqn" else "probability",
            "scores": [round(float(v), 4) for v in scores],
            "model": {"algo": self.algo, "run": self.run},
        }


def serve(agent: DispatchAgent, host: str, port: int):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "algo": agent.algo, "run": agent.run, "obs_dim": agent.obs_dim})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/recommend":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, agent.recommend(body["observation"]))
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})

    print(f"Serving {agent.algo} ({agent.run}) on http://{host}:{port}  [GET /health, POST /recommend]")
    HTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Serve a trained dispatch policy over HTTP")
    p.add_argument("--algo", default="dqn", choices=list(MODEL_PATHS))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    serve(DispatchAgent(a.algo), a.host, a.port)
