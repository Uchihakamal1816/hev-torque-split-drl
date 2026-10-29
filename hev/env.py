"""Gymnasium environment: torque-split control of a parallel HEV over a drive cycle.

Action: index into RATIOS. The engine is commanded to deliver RATIOS[a] times
the traction torque request. Values above 1 charge the battery through the
generator.

Cost (grams of fuel-equivalent per step, minimised):
    fuel burnt + battery degradation proxy + engine-start penalty
    + SOC-window penalty + unmet-torque penalty
and at the end of the episode, the change in battery energy is converted
to fuel equivalent, so draining the battery is not a free lunch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hev.cycles import ALL_PATTERNS, generate_cycle
from hev.vehicle import (
    G_FUEL_PER_KWH,
    GASOLINE_G_PER_L,
    VehicleParams,
    allocate,
    battery_power,
    battery_step,
    degradation_g,
    fuel_g,
    road_load,
)

RATIOS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
ICE_ONLY_INDEX = int(np.argmin(np.abs(RATIOS - 1.0)))

V_NORM = 33.3  # m/s, ~120 km/h
A_NORM = 2.0  # m/s^2
T_NORM = 200.0  # Nm
P_NORM = 50.0  # kW
OBS_DIM = 6
PREVIEW_S = 5

START_COST_G = 1.0  # fuel-equivalent cost of each engine start
SOC_WINDOW = (0.35, 0.75)
SOC_PENALTY_G_PER_UNIT = 400.0
UNMET_PENALTY_G_PER_NM = 5.0


@dataclass
class Totals:
    fuel_g: float = 0.0
    distance_m: float = 0.0
    ah: float = 0.0
    deg_g: float = 0.0
    engine_starts: int = 0
    engine_on_steps: int = 0
    unmet_nm: float = 0.0
    brake_nm: float = 0.0
    soc_violation_steps: int = 0
    trace: dict[str, list[float]] = field(default_factory=lambda: {k: [] for k in ("soc", "ice_on", "fuel_g")})


class HEVTorqueSplitEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        params: VehicleParams | None = None,
        patterns: tuple[str, ...] = ALL_PATTERNS,
        cycle_duration_s: int = 1200,
        dt: float = 1.0,
        reward_scale: float = 10.0,
    ):
        super().__init__()
        self.p = params or VehicleParams()
        self.patterns = patterns
        self.cycle_duration_s = cycle_duration_s
        self.dt = dt
        self.reward_scale = reward_scale
        self.action_space = spaces.Discrete(len(RATIOS))
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(OBS_DIM,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _load_cycle(self, speed: np.ndarray) -> None:
        accel = np.append(np.diff(speed) / self.dt, 0.0)
        self._speed = speed
        self._t_req, self._omega = road_load(speed, accel, self.p)
        self._power_kw = self._t_req * self._omega / 1000.0

    def _observe(self) -> np.ndarray:
        t = self._t
        speed = self._speed[t]
        accel = (self._speed[min(t + 1, len(self._speed) - 1)] - speed) / self.dt
        window = self._power_kw[t + 1 : t + 1 + PREVIEW_S]
        preview = float(window.mean()) if window.size else 0.0
        return np.array(
            [
                self._soc,
                speed / V_NORM,
                accel / A_NORM,
                self._t_req[t] / T_NORM,
                preview / P_NORM,
                float(self._engine_on_prev),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------ gym interface
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "speed" in options:
            speed = np.asarray(options["speed"], dtype=float)
        else:
            pattern = options.get("pattern", self.np_random.choice(self.patterns))
            speed = generate_cycle(pattern, self.cycle_duration_s, self.np_random, self.dt)
        self._load_cycle(speed)

        self._soc0 = float(options.get("soc0", self.np_random.uniform(0.45, 0.70)))
        self._soc = self._soc0
        self._t = 0
        self._engine_on_prev = False
        self.totals = Totals()
        return self._observe(), {}

    def step(self, action):
        t = self._t
        omega = self._omega[t]
        t_req = self._t_req[t]
        share = RATIOS[int(action)]

        split = allocate(t_req, omega, self._soc, share, self.p)
        p_batt = battery_power(split.t_em, omega, self.p)
        soc_new, current = battery_step(p_batt, self._soc, self.dt, self.p)

        fuel = fuel_g(split.t_ice, omega, self.dt, self.p)
        deg = degradation_g(current, self.dt, self.p)
        start = split.ice_on and not self._engine_on_prev
        violation = max(0.0, SOC_WINDOW[0] - soc_new) + max(0.0, soc_new - SOC_WINDOW[1])

        cost = (
            fuel
            + deg
            + START_COST_G * start
            + SOC_PENALTY_G_PER_UNIT * violation
            + UNMET_PENALTY_G_PER_NM * split.unmet
        )
        self._t += 1
        self._soc = soc_new
        last = self._t >= len(self._speed)
        if last:
            # Convert any change of battery energy into fuel-equivalent grams.
            delta_kwh = (self._soc0 - self._soc) * self.p.nominal_energy_kwh
            cost += delta_kwh * G_FUEL_PER_KWH

        tot = self.totals
        tot.fuel_g += fuel
        tot.distance_m += self._speed[t] * self.dt
        tot.ah += abs(current) * self.dt / 3600.0
        tot.deg_g += deg
        tot.engine_starts += int(start)
        tot.engine_on_steps += int(split.ice_on)
        tot.unmet_nm += split.unmet
        tot.brake_nm += split.brake
        tot.soc_violation_steps += int(violation > 0)
        tot.trace["soc"].append(soc_new)
        tot.trace["ice_on"].append(float(split.ice_on))
        tot.trace["fuel_g"].append(fuel)

        self._engine_on_prev = split.ice_on
        obs = self._observe() if not last else np.zeros(OBS_DIM, dtype=np.float32)
        reward = -cost / self.reward_scale
        info = {"soc": soc_new, "engine_on": split.ice_on, "fuel_g": fuel, "unmet_nm": split.unmet}
        return obs, float(reward), False, last, info

    # ---------------------------------------------------------------- summaries
    def summary(self) -> dict[str, float]:
        """Episode-level metrics, with the battery energy change folded into fuel."""
        tot = self.totals
        soc_end = self._soc
        fuel_corrected = tot.fuel_g + (self._soc0 - soc_end) * self.p.nominal_energy_kwh * G_FUEL_PER_KWH
        dist_km = tot.distance_m / 1000.0
        litres = fuel_corrected / GASOLINE_G_PER_L
        return {
            "distance_km": dist_km,
            "fuel_g": tot.fuel_g,
            "fuel_corrected_g": fuel_corrected,
            "l_per_100km": litres / dist_km * 100.0 if dist_km > 0 else float("nan"),
            "soc0": self._soc0,
            "soc_end": soc_end,
            "delta_soc": soc_end - self._soc0,
            "ah_throughput": tot.ah,
            "degradation_g": tot.deg_g,
            "engine_starts": tot.engine_starts,
            "engine_on_frac": tot.engine_on_steps / len(self._speed),
            "unmet_nm": tot.unmet_nm,
            "soc_violation_frac": tot.soc_violation_steps / len(self._speed),
        }
