"""Score candidate policies on the validation split (synthetic seeds 2000-2009).

All method and hyperparameter choices are made here, never on the test split.
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd
import torch
from stable_baselines3 import PPO

from hev.controllers import rule_based_action, with_soc_guard
from hev.evaluation import VAL_SEEDS, dp_factory, evaluate, static, synthetic_set
from hev.imitation import as_policy, make_net

METRICS = ["objective_l_eq_100km", "full_cost_l_eq_100km", "soc_violation_frac", "l_per_100km", "degradation_g", "engine_starts", "delta_soc", "unmet_nm"]


def load_il(path: pathlib.Path):
    net = make_net()
    net.load_state_dict(torch.load(path))
    return as_policy(net)


def load_ppo(path: pathlib.Path):
    model = PPO.load(path, device="cpu")
    return lambda obs: int(model.predict(obs, deterministic=True)[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--il", type=pathlib.Path, nargs="*", default=sorted(pathlib.Path("models/il").glob("*.pt")))
    ap.add_argument("--ppo", type=pathlib.Path, nargs="*", default=sorted(pathlib.Path("models/ppo").glob("*.zip")))
    ap.add_argument("--no-dp", action="store_true")
    ap.add_argument("--guard", action="store_true", help="also score imitation models with the SOC guard")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/validation.csv"))
    args = ap.parse_args()

    factories = {"Rule-based": static(rule_based_action)}
    if not args.no_dp:
        factories["DP (optimal, non-causal)"] = dp_factory
    factories.update({p.stem: static(load_il(p)) for p in args.il})
    if args.guard:
        factories.update({p.stem + "+guard": static(with_soc_guard(load_il(p))) for p in args.il})
    factories.update({p.stem: static(load_ppo(p)) for p in args.ppo})

    df = evaluate(factories, synthetic_set(VAL_SEEDS))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby("policy")[METRICS].mean().round(3).sort_values("objective_l_eq_100km").to_markdown())


if __name__ == "__main__":
    main()
