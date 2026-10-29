"""RL fine-tuning of the imitation policy with PPO (warm start from DAgger weights).

Phase 1 freezes the policy and trains only the value head, so the first
policy updates use sensible advantages. Phase 2 unfreezes the policy with a
small learning rate and clip range so RL refines, rather than overwrites,
what imitation learned.
"""

from __future__ import annotations

import argparse
import pathlib

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from torch import nn

from hev.datasets import TRAIN_REAL, load_many
from hev.env import HEVTorqueSplitEnv
from hev.imitation import make_net

VALUE_WARMUP_STEPS = 50_000


def load_warm_start(model: PPO, path: pathlib.Path) -> None:
    net = make_net()
    net.load_state_dict(torch.load(path))
    pi = model.policy.mlp_extractor.policy_net
    with torch.no_grad():
        pi[0].load_state_dict(net[0].state_dict())
        pi[2].load_state_dict(net[2].state_dict())
        model.policy.action_net.load_state_dict(net[4].state_dict())


def policy_params(model: PPO):
    return list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-cost", type=float, default=1.0)
    ap.add_argument("--timesteps", type=int, default=250_000)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    env = Monitor(HEVTorqueSplitEnv(fixed_cycles=load_many(TRAIN_REAL), p_fixed=0.3, start_cost=args.start_cost))
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-5, n_steps=4096, batch_size=256, n_epochs=5,
        gamma=0.999, gae_lambda=0.98, clip_range=0.1, ent_coef=0.0,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": nn.Tanh},
        seed=args.seed, device="cpu", verbose=0,
    )
    load_warm_start(model, args.init)

    for p in policy_params(model):
        p.requires_grad_(False)
    model.learn(total_timesteps=VALUE_WARMUP_STEPS)
    for p in policy_params(model):
        p.requires_grad_(True)
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out}.zip")


if __name__ == "__main__":
    main()
