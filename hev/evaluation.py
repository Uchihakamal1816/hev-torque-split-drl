"""Fixed evaluation splits and the episode runner shared by all scripts.

Splits (never mixed):
  train:      UDDS, HWFET, synthetic seeds 5000+   (train_imitation.py / train.py)
  validation: synthetic seeds 2000-2009            (all model and hyperparameter choices)
  test:       synthetic seeds 1000-1009 + WLTC 3a/3b (reported once, at the end)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from hev.cycles import generate_cycle
from hev.datasets import TEST_REAL, load_cycle
from hev.dp import DPSolution
from hev.env import HEVTorqueSplitEnv
from hev.vehicle import GASOLINE_G_PER_L

PATTERNS = ("urban", "suburban", "highway", "mixed")
VAL_SEEDS = range(2000, 2010)
TEST_SEEDS = range(1000, 1010)
SOC0 = 0.60

# A policy factory takes the cycle's speed trace and returns step(obs, env) -> action.
# Causal controllers ignore the speed trace; only the DP benchmark uses it.
Factory = Callable[[np.ndarray], Callable]


def static(policy: Callable[[np.ndarray], int]) -> Factory:
    return lambda speed: (lambda obs, env: policy(obs))


def dp_factory(speed: np.ndarray):
    sol = DPSolution(speed)
    return lambda obs, env: sol.best_action(env._t, env._soc, env._engine_on_prev)


def synthetic_set(seeds) -> list[tuple[str, str, np.ndarray]]:
    return [(f"{p}/{s}", p, generate_cycle(p, 1200, np.random.default_rng(s))) for p in PATTERNS for s in seeds]


def real_test_set() -> list[tuple[str, str, np.ndarray]]:
    return [(n, n, load_cycle(n)) for n in TEST_REAL]


def run_episode(env: HEVTorqueSplitEnv, factory: Factory, speed: np.ndarray, soc0: float = SOC0) -> dict:
    step = factory(speed)
    obs, _ = env.reset(options={"speed": speed, "soc0": soc0})
    done = False
    while not done:
        obs, _, _, done, _ = env.step(step(obs, env))
    s = env.summary()
    # Project objective: fuel (SOC-corrected) plus battery degradation, in litres-equivalent per 100 km.
    s["objective_l_eq_100km"] = (s["fuel_corrected_g"] + s["degradation_g"]) / GASOLINE_G_PER_L / s["distance_km"] * 100.0
    # Full cost: everything the controllers optimise (objective + start, SOC-window and unmet-torque penalties).
    s["full_cost_l_eq_100km"] = s["full_cost_g"] / GASOLINE_G_PER_L / s["distance_km"] * 100.0
    return s


def evaluate(factories: dict[str, Factory], cycles, env: HEVTorqueSplitEnv | None = None) -> pd.DataFrame:
    env = env or HEVTorqueSplitEnv()
    rows = []
    for cycle_id, group, speed in cycles:
        for name, factory in factories.items():
            rows.append({"cycle": cycle_id, "group": group, "policy": name, **run_episode(env, factory, speed)})
    return pd.DataFrame(rows)
