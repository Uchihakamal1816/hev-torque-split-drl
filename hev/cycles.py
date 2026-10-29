"""Synthetic drive cycles for stop-and-go, suburban and highway traffic.

Real driving cycles (WLTP, UDDS, ...) are fixed, so a policy can memorise
them. Here each episode draws a new speed trace from a traffic pattern,
which is what the "varying traffic patterns" requirement asks for.
Speeds are in m/s at 1 Hz.
"""

from __future__ import annotations

import numpy as np

PATTERNS: dict[str, dict] = {
    "urban": dict(
        v_range_kmh=(0.0, 50.0), stop_prob=0.30, dwell_s=(5, 30), segment_s=(10, 60),
        a_acc=(0.8, 1.6), a_dec=(0.9, 1.8),
    ),
    "suburban": dict(
        v_range_kmh=(30.0, 80.0), stop_prob=0.10, dwell_s=(0, 10), segment_s=(20, 90),
        a_acc=(0.6, 1.3), a_dec=(0.8, 1.5),
    ),
    "highway": dict(
        v_range_kmh=(90.0, 120.0), stop_prob=0.0, dwell_s=(0, 0), segment_s=(60, 240),
        a_acc=(0.4, 1.0), a_dec=(0.6, 1.2),
    ),
}
MIXED = "mixed"
ALL_PATTERNS = (*PATTERNS, MIXED)


def generate_cycle(pattern: str, duration_s: int, rng: np.random.Generator, dt: float = 1.0) -> np.ndarray:
    """Return a speed trace (m/s) of length `duration_s`.

    `pattern="mixed"` picks a new traffic regime for every segment, so one
    trace contains city stops, arterial cruising and motorway sections.
    """
    if pattern != MIXED and pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {pattern!r}; choose from {ALL_PATTERNS}")

    speed = np.zeros(duration_s)
    v = 0.0
    t = 0
    while t < duration_s:
        regime = pattern if pattern != MIXED else rng.choice(list(PATTERNS))
        cfg = PATTERNS[regime]

        stopping = rng.random() < cfg["stop_prob"]
        if stopping:
            target = 0.0
        else:
            lo, hi = cfg["v_range_kmh"]
            target = rng.uniform(lo, hi) / 3.6
        seg_len = int(rng.uniform(*cfg["segment_s"]))
        a_acc = rng.uniform(*cfg["a_acc"])
        a_dec = rng.uniform(*cfg["a_dec"])

        for _ in range(seg_len):
            if t >= duration_s:
                break
            accel = np.clip(0.3 * (target - v) + rng.normal(0.0, 0.05), -a_dec, a_acc)
            v = max(0.0, v + accel * dt)
            speed[t] = v
            t += 1

        if stopping:
            dwell = int(rng.uniform(*cfg["dwell_s"]))
            for _ in range(min(dwell, duration_s - t)):
                speed[t] = 0.0
                v = 0.0
                t += 1
    return speed
