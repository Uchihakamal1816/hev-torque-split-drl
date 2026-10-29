import numpy as np
import pytest

from hev.vehicle import VehicleParams, allocate, battery_power, bsfc_g_per_kwh, fuel_g, road_load

P = VehicleParams()


def test_road_load_zero_at_rest():
    t_req, omega = road_load(np.array([0.0]), np.array([0.0]), P)
    assert t_req[0] == pytest.approx(0.0)
    assert omega[0] == pytest.approx(0.0)


def test_road_load_positive_when_cruising_and_negative_when_braking():
    t_cruise, _ = road_load(np.array([20.0]), np.array([0.0]), P)
    t_brake, _ = road_load(np.array([20.0]), np.array([-2.0]), P)
    assert t_cruise[0] > 0
    assert t_brake[0] < 0


def test_engine_can_run_from_standstill_with_idle_speed_floor():
    split = allocate(t_req=80.0, omega=10.0, soc=0.6, ice_share=1.0, p=P)
    assert split.ice_on
    assert split.t_ice == pytest.approx(80.0)
    # fuel is charged at idle speed, not at the (near-zero) drivetrain speed
    assert fuel_g(split.t_ice, 10.0, 1.0, P) == pytest.approx(fuel_g(split.t_ice, P.ice_idle_speed, 1.0, P))


def test_split_balances_torque_exactly_when_feasible():
    split = allocate(t_req=60.0, omega=300.0, soc=0.6, ice_share=0.5, p=P)
    assert split.t_ice + split.t_em == pytest.approx(60.0)
    assert split.t_ice == pytest.approx(30.0)
    assert split.unmet == 0.0


def test_charging_share_above_one_pushes_motor_negative():
    split = allocate(t_req=40.0, omega=300.0, soc=0.5, ice_share=1.25, p=P)
    assert split.t_em < 0
    assert split.t_ice + split.t_em == pytest.approx(40.0)


def test_motor_saturation_falls_back_to_engine_and_reports_no_unmet():
    # Pure EV request above the motor's power-limited torque at 300 rad/s (~83 Nm):
    # the engine must pick up the difference, and the request is still met.
    split = allocate(t_req=150.0, omega=300.0, soc=0.6, ice_share=0.0, p=P)
    assert split.ice_on
    assert split.t_em <= 27_000 * P.em_eff / 300.0 + 1e-6
    assert split.t_ice + split.t_em == pytest.approx(150.0)
    assert split.unmet == 0.0


def test_infeasible_request_is_reported_as_unmet():
    split = allocate(t_req=200.0, omega=300.0, soc=0.6, ice_share=1.0, p=P)
    assert split.unmet > 0.0


def test_no_discharge_at_soc_floor():
    split = allocate(t_req=50.0, omega=300.0, soc=P.soc_min, ice_share=0.0, p=P)
    assert split.t_em <= 1e-9
    assert split.ice_on


def test_battery_power_sign_convention():
    assert battery_power(50.0, 300.0, P) > 0  # motoring draws power
    assert battery_power(-50.0, 300.0, P) < 0  # generating charges


def test_fuel_zero_without_engine_and_bsfc_best_near_sweet_spot():
    assert fuel_g(0.0, 300.0, 1.0, P) == 0.0
    assert bsfc_g_per_kwh(0.8 * P.ice_max_torque, 0.45 * P.ice_max_speed, P) == pytest.approx(230.0)
    assert bsfc_g_per_kwh(0.1 * P.ice_max_torque, 0.45 * P.ice_max_speed, P) > 230.0


def test_regen_is_limited_by_charge_power_and_excess_goes_to_friction_brake():
    # Hard braking at speed: far more regen torque than the 20 kW charge limit allows.
    split = allocate(t_req=-200.0, omega=300.0, soc=0.6, ice_share=0.0, p=P)
    charge_limit = -P.max_charge_kw * 1000.0 / (P.em_eff * 300.0)
    assert split.t_em == pytest.approx(charge_limit)
    assert split.brake == pytest.approx(charge_limit - (-200.0))


def test_no_regen_at_soc_ceiling():
    split = allocate(t_req=-50.0, omega=300.0, soc=P.soc_max, ice_share=0.0, p=P)
    assert split.t_em == 0.0
    assert split.brake == pytest.approx(50.0)


def test_battery_draw_never_exceeds_motor_limit_when_request_is_infeasible():
    split = allocate(t_req=400.0, omega=300.0, soc=0.6, ice_share=1.0, p=P)
    assert split.unmet > 0.0
    assert battery_power(split.t_em, 300.0, P) <= P.max_discharge_kw * 1000.0 + 1e-6
