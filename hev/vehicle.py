"""Quasi-static model of a parallel (P2-style) hybrid electric powertrain.

One fixed-ratio drivetrain, an engine and a motor that share the traction
torque, a battery modelled by its open-circuit voltage only, and a BSFC map
for fuel use. It is meant for controller research, not for calibration
against a specific production vehicle. Parameters are roughly Prius-class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

GRAVITY = 9.81  # m/s^2
AIR_DENSITY = 1.2  # kg/m^3
GASOLINE_G_PER_L = 745.0
GASOLINE_LHV_KWH_PER_KG = 12.33
ENGINE_NOMINAL_EFF = 0.30
# Fuel-equivalent cost of 1 kWh of battery energy, in grams of gasoline (~270 g/kWh).
G_FUEL_PER_KWH = 1000.0 / (GASOLINE_LHV_KWH_PER_KG * ENGINE_NOMINAL_EFF)


@dataclass(frozen=True)
class VehicleParams:
    # Chassis
    mass: float = 1400.0  # kg
    cd: float = 0.28
    frontal_area: float = 2.2  # m^2
    crr: float = 0.012
    wheel_radius: float = 0.31  # m
    final_drive: float = 4.1  # fixed overall ratio, single speed
    drive_eff: float = 0.96
    # Engine
    ice_max_torque: float = 115.0  # Nm
    ice_max_speed: float = 500.0  # rad/s
    ice_idle_speed: float = 75.0  # rad/s, engine speed floor (slipping clutch below this)
    ice_min_torque: float = 8.0  # Nm, smaller requests switch the engine off
    # Motor / generator
    em_max_torque: float = 160.0  # Nm
    em_eff: float = 0.92
    # Battery (Prius-class NiMH pack)
    capacity_ah: float = 6.5
    nominal_energy_kwh: float = 1.31
    max_discharge_kw: float = 27.0
    max_charge_kw: float = 20.0
    soc_min: float = 0.30  # hard limits, motoring/generation is cut off outside these
    soc_max: float = 0.80
    # Battery degradation proxy: grams fuel-equivalent per Ah of throughput,
    # amplified at high current (C-rate stress), with i_ref as the knee.
    deg_g_per_ah: float = 20.0
    i_ref_a: float = 40.0


class Split(NamedTuple):
    t_ice: float  # engine torque, Nm
    t_em: float  # motor/generator torque, Nm (negative = charging)
    ice_on: bool
    unmet: float  # traction torque the powertrain could not deliver, Nm
    brake: float  # regenerative torque that could not be absorbed (friction brake), Nm


def open_circuit_voltage(soc: float) -> float:
    return 180.0 + 45.0 * soc


def road_load(speed: np.ndarray, accel: np.ndarray, p: VehicleParams) -> tuple[np.ndarray, np.ndarray]:
    """Motor-side torque request (Nm, negative when braking) and drivetrain speed (rad/s)."""
    inertial = p.mass * accel
    rolling = np.sign(speed) * p.mass * GRAVITY * p.crr
    drag = 0.5 * AIR_DENSITY * p.cd * p.frontal_area * speed * np.abs(speed)
    force = inertial + rolling + drag
    t_req = force * p.wheel_radius / (p.final_drive * p.drive_eff)
    omega = p.final_drive * speed / p.wheel_radius
    return t_req, omega


def emotor_limits(omega: float, soc: float, p: VehicleParams) -> tuple[float, float]:
    """Return (lowest, highest) allowed motor torque at this speed and SOC."""
    w = max(omega, 1.0)
    hi = min(p.em_max_torque, p.max_discharge_kw * 1000.0 * p.em_eff / w)
    lo = -min(p.em_max_torque, p.max_charge_kw * 1000.0 / (p.em_eff * w))
    if soc <= p.soc_min:
        hi = 0.0
    if soc >= p.soc_max:
        lo = 0.0
    return lo, hi


def allocate(t_req: float, omega: float, soc: float, ice_share: float, p: VehicleParams) -> Split:
    """Split the traction torque between engine and motor.

    `ice_share` is the commanded engine torque as a multiple of the traction
    request. Values above 1 push the surplus through the generator to charge
    the battery. The motor is the balancing element: whatever the engine does
    not deliver, the motor covers, and when the motor saturates the engine
    picks up the rest (supervisory fallback), so the requested split is only
    ever a suggestion, not a guarantee.
    """
    # The engine can run from standstill: a clutch (or planetary) slips at low
    # vehicle speed, so the engine speed is floored at idle in fuel_g().
    ice_available = t_req > 0.0
    t_ice = min(max(ice_share * t_req, 0.0), p.ice_max_torque) if ice_available else 0.0
    if t_ice < p.ice_min_torque:
        t_ice = 0.0

    lo, hi = emotor_limits(omega, soc, p)
    t_em = float(np.clip(t_req - t_ice, lo, hi))
    if ice_available:
        t_ice = float(np.clip(t_req - t_em, 0.0, p.ice_max_torque))
        if t_ice < p.ice_min_torque:
            t_ice = 0.0
    # The motor can only deliver what its limits allow. Traction it cannot
    # cover is unmet; braking it cannot absorb goes to the friction brakes.
    t_em_needed = t_req - t_ice
    t_em = min(max(t_em_needed, lo), hi)
    unmet = max(0.0, t_em_needed - hi) if t_req > 0 else 0.0
    brake = max(0.0, lo - t_em_needed) if t_req < 0 else 0.0
    return Split(t_ice=t_ice, t_em=t_em, ice_on=t_ice > 0.0, unmet=unmet, brake=brake)


def battery_power(t_em: float, omega: float, p: VehicleParams) -> float:
    """Electrical power at the battery terminals in W (positive = discharge)."""
    mech = t_em * omega
    return mech / p.em_eff if mech >= 0 else mech * p.em_eff


def bsfc_g_per_kwh(t_ice: float, omega: float, p: VehicleParams) -> float:
    """Brake specific fuel consumption, a quadratic bowl around the efficient island."""
    load = t_ice / p.ice_max_torque
    speed = omega / p.ice_max_speed
    return 230.0 * (
        1.0
        + 0.6 * (load - 0.8) ** 2
        + 0.3 * (speed - 0.45) ** 2
        + 0.5 * max(0.0, 0.25 - load)
    )


def fuel_g(t_ice: float, omega: float, dt: float, p: VehicleParams) -> float:
    if t_ice <= 0.0:
        return 0.0
    omega_ice = max(omega, p.ice_idle_speed)
    power_kw = t_ice * omega_ice / 1000.0
    return bsfc_g_per_kwh(t_ice, omega_ice, p) * power_kw * dt / 3600.0


def battery_step(p_batt: float, soc: float, dt: float, p: VehicleParams) -> tuple[float, float]:
    """Advance SOC by one step. Returns (new_soc, current_A)."""
    current = p_batt / open_circuit_voltage(soc)
    new_soc = soc - current * dt / (3600.0 * p.capacity_ah)
    return float(np.clip(new_soc, 0.0, 1.0)), current


def degradation_g(current: float, dt: float, p: VehicleParams) -> float:
    """Throughput-based degradation proxy in grams fuel-equivalent."""
    amp = abs(current)
    throughput_ah = amp * dt / 3600.0
    return p.deg_g_per_ah * throughput_ah * (1.0 + (amp / p.i_ref_a) ** 2)
