"""Compare ICE-only, rule-based and DQN torque-split policies on held-out cycles.

Two held-out sets:
  synthetic: 4 traffic patterns x 10 unseen seeds (generated, never trained on)
  real:      WLTC class 3a and 3b (published cycles, never trained on by any model)
"""

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
from hev.datasets import TEST_REAL, load_cycle  # noqa: E402
from hev.env import HEVTorqueSplitEnv  # noqa: E402

PATTERNS = ("urban", "suburban", "highway", "mixed")
EVAL_SEEDS = range(1000, 1010)  # disjoint from training seeds
SOC0 = 0.60
METRICS = ["l_per_100km", "delta_soc", "engine_starts", "engine_on_frac", "unmet_nm", "degradation_g"]


def run_episode(env, policy, speed: np.ndarray, soc0: float):
    obs, _ = env.reset(options={"speed": speed, "soc0": soc0})
    done = False
    while not done:
        obs, _, _, done, _ = env.step(policy(obs))
    return env.summary(), env.totals.trace


def build_policies(args) -> dict:
    policies = {"ICE only": ice_only_action, "Rule-based": rule_based_action}
    for label, path in (("DQN synthetic", args.synthetic_model), ("DQN real", args.real_model)):
        if path.exists():
            model = DQN.load(path, device="cpu")
            policies[label] = (lambda m: lambda obs: int(m.predict(obs, deterministic=True)[0]))(model)
        else:
            print(f"[warn] {path} not found, skipping {label}")
    return policies


def evaluate_set(env, policies, cycles, label_of, traces=None):
    rows = []
    for key, speed in cycles:
        for name, policy in policies.items():
            summary, trace = run_episode(env, policy, speed, SOC0)
            rows.append({"cycle": label_of(key), "policy": name, **summary})
            if traces is not None and key in traces["keys"]:
                traces["data"][(label_of(key), name)] = trace
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic-model", type=pathlib.Path, default=pathlib.Path("models/dqn_synthetic.zip"))
    ap.add_argument("--real-model", type=pathlib.Path, default=pathlib.Path("models/dqn_real.zip"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    policies = build_policies(args)
    env = HEVTorqueSplitEnv()

    # Synthetic held-out set
    syn_cycles = [((p, s), generate_cycle(p, env.cycle_duration_s, np.random.default_rng(s)))
                  for p in PATTERNS for s in EVAL_SEEDS]
    syn_df = evaluate_set(env, policies, syn_cycles, lambda k: k[0])
    syn_df["seed"] = [s for p in PATTERNS for s in EVAL_SEEDS for _ in policies]
    syn_df.to_csv(args.out / "per_cycle.csv", index=False)
    syn_table = syn_df.groupby(["cycle", "policy"])[METRICS].mean().round(3)
    syn_table.to_csv(args.out / "summary_by_pattern.csv")
    syn_overall = syn_df.groupby("policy")[METRICS].mean().round(3)
    syn_overall.to_csv(args.out / "summary_overall.csv")

    # Real held-out set (WLTC class 3a and 3b)
    real_cycles = [(n, load_cycle(n)) for n in TEST_REAL]
    traces = {"keys": {n for n in TEST_REAL}, "data": {}}
    real_df = evaluate_set(env, policies, real_cycles, lambda k: k, traces)
    real_df.to_csv(args.out / "real_cycles.csv", index=False)
    real_overall = real_df.groupby("policy")[METRICS].mean().round(3)
    real_overall.to_csv(args.out / "real_summary_overall.csv")

    print("\nSynthetic held-out, mean over 40 cycles (L/100 km is SOC-corrected):")
    print(syn_overall.to_markdown())
    print("\nSynthetic held-out, by pattern:")
    print(syn_table.to_markdown())
    print("\nReal held-out (WLTC 3a + 3b):")
    print(real_df[["cycle", "policy", "l_per_100km", "delta_soc", "engine_starts", "engine_on_frac", "unmet_nm"]].round(3).to_markdown(index=False))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for name in policies:
        tr = traces["data"].get(("wltc_class3b", name))
        if tr is None:
            continue
        axes[0].plot(tr["soc"], label=name)
        axes[1].plot(np.cumsum(tr["fuel_g"]), label=name)
    axes[0].set_ylabel("SOC")
    axes[1].set_ylabel("cumulative fuel (g)")
    axes[1].set_xlabel("time (s)")
    axes[0].legend()
    axes[0].set_title("WLTC class 3b (held out)")
    fig.tight_layout()
    fig.savefig(args.out / "wltc3b_soc_fuel.png", dpi=120)
    print(f"\nwrote tables and plots to {args.out}/")


if __name__ == "__main__":
    main()
