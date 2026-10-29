"""Train a DQN torque-split agent on randomised traffic cycles."""

from __future__ import annotations

import argparse
import pathlib

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from hev.env import HEVTorqueSplitEnv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("models/dqn_hev"))
    args = ap.parse_args()

    env = Monitor(HEVTorqueSplitEnv())
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        device="cpu",  # small MLP; avoids CUDA contention
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out}.zip")


if __name__ == "__main__":
    main()
