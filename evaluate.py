"""Final, one-shot evaluation on the TEST split.

Test split: synthetic seeds 1000-1009 (4 patterns, 40 cycles) and the real
WLTC class 3a and 3b cycles. No model or hyperparameter was chosen on these
cycles; choices were made on the validation split (validate.py).

Multi-seed methods are reported as mean ± std across training seeds.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402

from hev.controllers import ice_only_action, rule_based_action, with_soc_guard  # noqa: E402
from hev.datasets import load_cycle  # noqa: E402
from hev.env import HEVTorqueSplitEnv  # noqa: E402
from hev.evaluation import TEST_SEEDS, dp_factory, evaluate, real_test_set, run_episode, static, synthetic_set  # noqa: E402
from validate import load_il, load_ppo  # noqa: E402

METRICS = ["objective_l_eq_100km", "full_cost_l_eq_100km", "soc_violation_frac", "l_per_100km", "degradation_g", "engine_starts", "delta_soc", "unmet_nm"]
SEED_RE = re.compile(r"^(.*) \[seed(\d+)\]$")


def factories(args) -> dict:
    f = {"ICE only": static(ice_only_action), "Rule-based": static(rule_based_action)}
    if args.dqn.exists():
        dqn = DQN.load(args.dqn, device="cpu")
        f["DQN (from scratch)"] = static(lambda obs: int(dqn.predict(obs, deterministic=True)[0]))
    for p in sorted(pathlib.Path("models/il").glob(f"il_sc{args.start_cost}_seed*.pt")):
        f[f"Imitation of DP [seed{p.stem[-1]}]"] = static(load_il(p))
        f[f"Imitation of DP + SOC guard [seed{p.stem[-1]}]"] = static(with_soc_guard(load_il(p)))
    for p in sorted(pathlib.Path("models/ppo").glob(f"ilppo_sc{args.start_cost}_seed*.zip")):
        f[f"Imitation + PPO [seed{p.stem[-1]}]"] = static(load_ppo(p))
    f["DP (optimal, non-causal)"] = dp_factory
    return f


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df["policy"].str.extract(SEED_RE.pattern)
    df["method"] = m[0].fillna(df["policy"])
    df["seed"] = m[1].fillna("-")
    per_seed = df.groupby(["method", "seed"])[METRICS].mean()
    mean = per_seed.groupby("method").mean()
    std = per_seed.groupby("method").std().fillna(0.0)
    out = mean.round(3).astype(str)
    for col in METRICS:
        multi = per_seed.groupby("method").size() > 1
        out.loc[multi, col] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(mean.loc[multi, col], std.loc[multi, col])]
    return out.loc[mean.sort_values("objective_l_eq_100km").index]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-cost", type=int, default=1, help="start-cost variant selected on validation")
    ap.add_argument("--dqn", type=pathlib.Path, default=pathlib.Path("models/dqn_real.zip"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    f = factories(args)
    syn = evaluate(f, synthetic_set(TEST_SEEDS))
    real = evaluate(f, real_test_set())
    syn.to_csv(args.out / "test_synthetic_per_cycle.csv", index=False)
    real.to_csv(args.out / "test_wltc_per_cycle.csv", index=False)

    syn_sum, real_sum = summarise(syn), summarise(real)
    syn_sum.to_csv(args.out / "test_synthetic_summary.csv")
    real_sum.to_csv(args.out / "test_wltc_summary.csv")
    print("TEST: synthetic traffic (40 unseen cycles). Objective = fuel + degradation, L-eq/100 km\n")
    print(syn_sum.to_markdown())
    print("\nTEST: real WLTC class 3a + 3b (never trained on)\n")
    print(real_sum.to_markdown())

    by_pattern = syn.assign(method=syn["policy"].str.replace(r" \[seed\d+\]$", "", regex=True))
    bp = by_pattern.groupby(["group", "method"])[["objective_l_eq_100km", "l_per_100km"]].mean().round(3)
    bp.to_csv(args.out / "test_synthetic_by_pattern.csv")

    # SOC and cumulative fuel on WLTC 3b for the main controllers.
    env = HEVTorqueSplitEnv()
    speed = load_cycle("wltc_class3b")
    show = ["Rule-based", "Imitation of DP [seed0]", "DP (optimal, non-causal)"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for name in show:
        if name not in f:
            continue
        run_episode(env, f[name], speed)
        axes[0].plot(env.totals.trace["soc"], label=name.replace(" [seed0]", ""))
        axes[1].plot(np.cumsum(env.totals.trace["fuel_g"]), label=name.replace(" [seed0]", ""))
    axes[0].set_ylabel("SOC")
    axes[1].set_ylabel("cumulative fuel (g)")
    axes[1].set_xlabel("time (s)")
    axes[0].legend()
    axes[0].set_title("WLTC class 3b (held-out test cycle)")
    fig.tight_layout()
    fig.savefig(args.out / "wltc3b_soc_fuel.png", dpi=120)
    print(f"\nwrote test tables and plot to {args.out}/")


if __name__ == "__main__":
    main()
