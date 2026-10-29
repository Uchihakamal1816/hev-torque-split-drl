"""Compare ICE-only, rule-based and DQN torque-split policies on held-out cycles."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402

from hev.controllers import ice_only_action, rule_based_action  # noqa: E402
from hev.cycles import generate_cycle  # noqa: E402
from hev.env import HEVTorqueSplitEnv  # noqa: E402

PATTERNS = ("urban", "suburban", "highway", "mixed")
EVAL_SEEDS = range(1000, 1010)  # disjoint from training seeds
SOC0 = 0.60


def run_episode(env, policy, speed: np.ndarray, soc0: float):
    obs, _ = env.reset(options={"speed": speed, "soc0": soc0})
    done = False
    while not done:
        action = policy(obs)
        obs, _, _, done, _ = env.step(action)
    return env.summary(), env.totals.trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=pathlib.Path, default=pathlib.Path("models/dqn_hev.zip"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    policies = {"ICE only": None, "Rule-based": rule_based_action}
    if args.model.exists():
        dqn = DQN.load(args.model, device="cpu")
        policies["DQN"] = lambda obs: int(dqn.predict(obs, deterministic=True)[0])
    else:
        print(f"[warn] {args.model} not found, skipping DQN")

    env = HEVTorqueSplitEnv()
    rows, traces = [], {}
    for pattern in PATTERNS:
        for seed in EVAL_SEEDS:
            speed = generate_cycle(pattern, env.cycle_duration_s, np.random.default_rng(seed))
            for name, pol in policies.items():
                policy = pol if pol is not None else ice_only_action
                summary, trace = run_episode(env, policy, speed, SOC0)
                rows.append({"pattern": pattern, "seed": seed, "policy": name, **summary})
                if seed == EVAL_SEEDS[0]:
                    traces[(pattern, name)] = trace

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "per_cycle.csv", index=False)

    metrics = ["l_per_100km", "delta_soc", "engine_starts", "engine_on_frac", "unmet_nm", "degradation_g"]
    table = df.groupby(["pattern", "policy"])[metrics].mean().round(3)
    table.to_csv(args.out / "summary_by_pattern.csv")
    overall = df.groupby("policy")[metrics].mean().round(3)
    overall.to_csv(args.out / "summary_overall.csv")

    print("\nMean over all held-out cycles (L/100 km is SOC-corrected fuel):")
    print(overall.to_markdown())
    print("\nBy traffic pattern:")
    print(table.to_markdown())

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for name in policies:
        tr = traces[("mixed", name)]
        axes[0].plot(tr["soc"], label=name)
        axes[1].plot(np.cumsum(tr["fuel_g"]), label=name)
    axes[0].set_ylabel("SOC")
    axes[1].set_ylabel("cumulative fuel (g)")
    axes[1].set_xlabel("time (s)")
    axes[0].legend()
    axes[0].set_title("Mixed-traffic cycle, held-out seed")
    fig.tight_layout()
    fig.savefig(args.out / "mixed_cycle_soc_fuel.png", dpi=120)
    print(f"\nwrote tables and plot to {args.out}/")


if __name__ == "__main__":
    main()
