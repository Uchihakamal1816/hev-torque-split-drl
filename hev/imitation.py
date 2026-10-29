"""Learn a causal torque-split policy by imitating the DP expert (DAgger).

DP needs the whole future speed trace, so it cannot run in a vehicle. DAgger
trains a network that maps the same 6-dim observation the rule-based
controller sees to the action DP would choose, on states the network itself
visits, so it learns to recover from its own mistakes.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn

from hev.dp import DPSolution
from hev.env import OBS_DIM, RATIOS, HEVTorqueSplitEnv

HIDDEN = 128


def make_net() -> nn.Sequential:
    # Same shape as Stable-Baselines3's PPO policy head (Tanh MLP), so the
    # weights can be copied into PPO for RL fine-tuning.
    return nn.Sequential(
        nn.Linear(OBS_DIM, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, len(RATIOS)),
    )


def as_policy(net: nn.Module) -> Callable[[np.ndarray], int]:
    net.eval()

    def policy(obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(net(torch.as_tensor(obs, dtype=torch.float32)).argmax())

    return policy


def labelled_rollout(env, speed, sol: DPSolution, soc0: float, actor=None):
    """Drive the cycle with `actor` (or the expert if None); label every state with DP's action."""
    obs, _ = env.reset(options={"speed": speed, "soc0": soc0})
    xs, ys, done = [], [], False
    while not done:
        expert = sol.best_action(env._t, env._soc, env._engine_on_prev)
        xs.append(obs)
        ys.append(expert)
        obs, _, _, done, _ = env.step(expert if actor is None else actor(obs))
    return xs, ys


def fit(net: nn.Module, X: np.ndarray, Y: np.ndarray, epochs: int, seed: int, lr: float = 1e-3) -> None:
    g = torch.Generator().manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.long)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 1024):
            idx = perm[i : i + 1024]
            opt.zero_grad()
            loss_fn(net(Xt[idx]), Yt[idx]).backward()
            opt.step()


def dagger(cycles: list[np.ndarray], *, seed: int = 0, start_cost: float = 1.0, iters: int = 4,
           epochs: int = 15, soc0s=(0.45, 0.55, 0.65), log=print) -> nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = HEVTorqueSplitEnv(start_cost=start_cost)
    sols = [DPSolution(c, start_cost=start_cost) for c in cycles]
    log(f"solved DP on {len(sols)} training cycles")

    X, Y = [], []
    for c, sol in zip(cycles, sols):
        for soc0 in soc0s:
            xs, ys = labelled_rollout(env, c, sol, soc0)
            X += xs
            Y += ys
    net = make_net()
    for it in range(iters):
        fit(net, np.array(X), np.array(Y), epochs, seed + it)
        actor = as_policy(net)
        agree = []
        for c, sol in zip(cycles, sols):
            xs, ys = labelled_rollout(env, c, sol, float(rng.uniform(0.45, 0.70)), actor)
            X += xs
            Y += ys
            with torch.no_grad():
                pred = net(torch.as_tensor(np.array(xs), dtype=torch.float32)).argmax(1).numpy()
            agree.append(np.mean(pred == np.array(ys)))
        log(f"DAgger iter {it + 1}/{iters}: {len(X)} samples, on-policy agreement with DP {np.mean(agree):.3f}")
    fit(net, np.array(X), np.array(Y), epochs, seed + iters)
    return net
