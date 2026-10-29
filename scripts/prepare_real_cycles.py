"""Convert the raw published drive-cycle files into speed traces in m/s.

EPA UDDS / HWFET files list speed in mph, one row per second.
WLTC files are whitespace-separated speeds in km/h, one per second.
Output: data/cycles/<name>.csv with columns time_s, speed_mps.
"""

from __future__ import annotations

import pathlib

import numpy as np

RAW = pathlib.Path("data/raw")
OUT = pathlib.Path("data/cycles")
MPH_TO_MPS = 0.44704
KMH_TO_MPS = 1.0 / 3.6


def read_epa(path: pathlib.Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            rows.append(float(parts[1]))
    return np.array(rows) * MPH_TO_MPS


def read_wltc(path: pathlib.Path) -> np.ndarray:
    values = [float(x) for x in path.read_text().split()]
    return np.array(values) * KMH_TO_MPS


def write(name: str, speed: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(speed))
    data = np.column_stack([t, speed])
    np.savetxt(OUT / f"{name}.csv", data, delimiter=",", header="time_s,speed_mps", comments="", fmt=["%d", "%.4f"])
    print(f"{name}: {len(speed)} s, max {speed.max() * 3.6:.1f} km/h")


def main() -> None:
    write("udds", read_epa(RAW / "uddscol.txt"))
    write("hwfet", read_epa(RAW / "hwycol.txt"))
    write("wltc_class3a", read_wltc(RAW / "V_class3a.txt")[:1800])
    write("wltc_class3b", read_wltc(RAW / "V_class3b.txt")[:1800])


if __name__ == "__main__":
    main()
