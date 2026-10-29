"""Dynamic-programming benchmark: the cost-optimal torque split for a known cycle.

DP sees the whole speed trace in advance, so it is not a causal controller.
It gives the lowest cost any controller could reach under this model (the
standard upper bound in HEV energy management), and it labels optimal
actions for imitation learning.

State: (SOC on a grid, engine on/off at the previous step).
Cost: the same per-step cost as HEVTorqueSplitEnv, plus the terminal
battery-energy correction, so DP optimises exactly what the env rewards.
"""

from __future__ import annotations

import numpy as np

from hev.env import RATIOS, SOC_PENALTY_G_PER_UNIT, SOC_WINDOW, START_COST_G, UNMET_PENALTY_G_PER_NM
from hev.vehicle import G_FUEL_PER_KWH, VehicleParams, open_circuit_voltage, road_load

SOC_GRID = np.linspace(0.28, 0.82, 217)  # 0.0025 SOC resolution


def _fuel_vec(t_ice: np.ndarray, omega: float, dt: float, p: VehicleParams) -> np.ndarray:
    omega_ice = max(omega, p.ice_idle_speed)
    load = t_ice / p.ice_max_torque
    speed = omega_ice / p.ice_max_speed
    bsfc = 230.0 * (1.0 + 0.6 * (load - 0.8) ** 2 + 0.3 * (speed - 0.45) ** 2 + 0.5 * np.maximum(0.0, 0.25 - load))
    return np.where(t_ice > 0.0, bsfc * t_ice * omega_ice / 1000.0 * dt / 3600.0, 0.0)


def step_all_actions(
    t_req: float, omega: float, soc: np.ndarray, engine_prev: np.ndarray,
    p: VehicleParams, dt: float, start_cost: float,
):
    """One model step for every action, vectorised over states.

    Mirrors vehicle.allocate / battery_step / fuel_g / degradation_g and the
    env cost exactly. Returns arrays of shape (n_actions, n_states):
    soc_next, ice_on, cost.
    """
    w = max(omega, 1.0)
    hi = np.where(soc <= p.soc_min, 0.0, min(p.em_max_torque, p.max_discharge_kw * 1000.0 * p.em_eff / w))
    lo = np.where(soc >= p.soc_max, 0.0, -min(p.em_max_torque, p.max_charge_kw * 1000.0 / (p.em_eff * w)))
    available = t_req > 0.0
    zeros = np.zeros_like(soc)
    ocv = open_circuit_voltage(soc)

    soc_out, on_out, cost_out = [], [], []
    for share in RATIOS:
        t_ice = min(max(share * t_req, 0.0), p.ice_max_torque) if available else 0.0
        if t_ice < p.ice_min_torque:
            t_ice = 0.0
        t_em = np.clip(t_req - t_ice, lo, hi)
        if available:
            t_ice = np.clip(t_req - t_em, 0.0, p.ice_max_torque)
            t_ice = np.where(t_ice < p.ice_min_torque, 0.0, t_ice)
        else:
            t_ice = zeros
        t_em_needed = t_req - t_ice
        unmet = np.maximum(0.0, t_em_needed - hi) if t_req > 0 else zeros
        t_em = np.clip(t_em_needed, lo, hi)

        mech = t_em * omega
        p_batt = np.where(mech >= 0, mech / p.em_eff, mech * p.em_eff)
        current = p_batt / ocv
        soc_next = np.clip(soc - current * dt / (3600.0 * p.capacity_ah), 0.0, 1.0)

        amp = np.abs(current)
        deg = p.deg_g_per_ah * amp * dt / 3600.0 * (1.0 + (amp / p.i_ref_a) ** 2)
        ice_on = t_ice > 0.0
        start = ice_on & ~engine_prev
        violation = np.maximum(0.0, SOC_WINDOW[0] - soc_next) + np.maximum(0.0, soc_next - SOC_WINDOW[1])
        cost = (_fuel_vec(t_ice, omega, dt, p) + deg + start_cost * start
                + SOC_PENALTY_G_PER_UNIT * violation + UNMET_PENALTY_G_PER_NM * unmet)
        soc_out.append(soc_next)
        on_out.append(ice_on)
        cost_out.append(cost)
    return np.array(soc_out), np.array(on_out), np.array(cost_out)


class DPSolution:
    """Backward DP over one speed trace; query optimal actions for any state."""

    def __init__(self, speed: np.ndarray, p: VehicleParams | None = None, dt: float = 1.0,
                 start_cost: float = START_COST_G, grid: np.ndarray = SOC_GRID):
        self.p = p or VehicleParams()
        self.dt = dt
        self.start_cost = start_cost
        self.grid = grid
        accel = np.append(np.diff(speed) / dt, 0.0)
        self.t_req, self.omega = road_load(np.asarray(speed, float), accel, self.p)
        T, G = len(speed), len(grid)

        # Both engine states stacked: rows 0..G-1 engine off, G..2G-1 engine on.
        soc = np.tile(grid, 2)
        eprev = np.repeat([False, True], G)
        V = np.empty((T + 1, 2 * G))
        # Terminal: battery energy converted to fuel; constant terms dropped.
        V[T] = -soc * self.p.nominal_energy_kwh * G_FUEL_PER_KWH
        for t in range(T - 1, -1, -1):
            V[t] = self._q(t, soc, eprev, V[t + 1]).min(axis=0)
        self.V = V

    def _q(self, t: int, soc: np.ndarray, eprev: np.ndarray, v_next: np.ndarray) -> np.ndarray:
        G = len(self.grid)
        soc_next, on, cost = step_all_actions(self.t_req[t], self.omega[t], soc, eprev, self.p, self.dt, self.start_cost)
        v_off = np.interp(soc_next, self.grid, v_next[:G])
        v_on = np.interp(soc_next, self.grid, v_next[G:])
        return cost + np.where(on, v_on, v_off)

    def best_action(self, t: int, soc: float, engine_prev: bool) -> int:
        q = self._q(t, np.array([soc]), np.array([bool(engine_prev)]), self.V[t + 1])
        return int(np.argmin(q[:, 0]))
