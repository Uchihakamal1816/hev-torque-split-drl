"""Train the imitation (DAgger-from-DP) torque-split policy."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from hev.cycles import PATTERNS, generate_cycle
from hev.datasets import TRAIN_REAL, load_many
from hev.imitation import dagger

TRAIN_SEEDS = range(5000, 5015)  # per pattern; disjoint from validation (2000+) and test (1000+)


def training_cycles(duration_s: int = 1200) -> list[np.ndarray]:
    cycles = load_many(TRAIN_REAL)
    for pattern in (*PATTERNS, "mixed"):
        cycles += [generate_cycle(pattern, duration_s, np.random.default_rng(s)) for s in TRAIN_SEEDS]
    return cycles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-cost", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    net = dagger(training_cycles(), seed=args.seed, start_cost=args.start_cost, iters=args.iters)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
