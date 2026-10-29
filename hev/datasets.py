"""Loaders for the real published drive cycles in data/cycles/."""

from __future__ import annotations

import pathlib

import numpy as np

CYCLE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "cycles"
TRAIN_REAL = ("udds", "hwfet")  # used for training when real data is enabled
TEST_REAL = ("wltc_class3a", "wltc_class3b")  # never used in training


def load_cycle(name: str) -> np.ndarray:
    """Speed trace in m/s, one sample per second."""
    data = np.loadtxt(CYCLE_DIR / f"{name}.csv", delimiter=",", skiprows=1)
    return data[:, 1]


def load_many(names: tuple[str, ...]) -> list[np.ndarray]:
    return [load_cycle(n) for n in names]
