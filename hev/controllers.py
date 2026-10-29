"""Hand-written baselines to compare the learned policy against."""

from __future__ import annotations

import numpy as np

from hev.env import ICE_ONLY_INDEX, RATIOS, T_NORM

RB_LOW_SOC = 0.45
RB_HIGH_SOC = 0.65
RB_EV_TORQUE_NM = 40.0  # switch the engine on above this request
RB_EV_OFF_TORQUE_NM = 25.0  # switch it back off only below this (hysteresis)
RB_CHARGE_TORQUE_NM = 60.0
RB_ASSIST_TORQUE_NM = 110.0


def rule_based_action(obs: np.ndarray) -> int:
    """Thermostat-style state machine: charge when low, electric when light or full.

    The engine has hysteresis on the torque request so it does not chatter
    on and off around a single threshold. Reads the same observation vector
    the RL agent sees, so the two controllers are compared on identical information.
    """
    soc = float(obs[0])
    t_req = float(obs[3]) * T_NORM
    engine_was_on = float(obs[5]) > 0.5
    ev_threshold = RB_EV_OFF_TORQUE_NM if engine_was_on else RB_EV_TORQUE_NM

    if t_req <= 0:
        ratio = 0.0  # regenerative braking, engine off
    elif soc < RB_LOW_SOC:
        ratio = 1.25 if t_req < RB_CHARGE_TORQUE_NM else 1.0  # charge while driving
    elif soc > RB_HIGH_SOC or t_req < ev_threshold:
        ratio = 0.0  # electric drive
    elif t_req > RB_ASSIST_TORQUE_NM:
        ratio = 0.75  # heavy load: engine does most, motor assists
    else:
        ratio = 1.0
    return int(np.argmin(np.abs(RATIOS - ratio)))


def ice_only_action(obs: np.ndarray) -> int:
    return ICE_ONLY_INDEX


# SOC safety guard thresholds, set from the 0.35-0.75 SOC window with a margin (not tuned).
GUARD_NO_CHARGE_SOC = 0.70  # above this, never command battery charging
GUARD_MOTOR_LED_SOC = 0.72  # above this, the motor leads (engine share at most 0.25)
GUARD_NO_ASSIST_SOC = 0.38  # below this, the engine covers the full request


def with_soc_guard(policy):
    """Wrap a learned policy with a rule-based shield that keeps SOC inside its window.

    The learned policy proposes an action; the guard only overrides it near
    the edges of the SOC window, where battery health matters most.
    """

    def guarded(obs: np.ndarray) -> int:
        action = policy(obs)
        soc = float(obs[0])
        t_req = float(obs[3]) * T_NORM
        ratio = float(RATIOS[action])
        if t_req > 0:
            if soc > GUARD_MOTOR_LED_SOC:
                ratio = min(ratio, 0.25)
            elif soc > GUARD_NO_CHARGE_SOC:
                ratio = min(ratio, 1.0)
            elif soc < GUARD_NO_ASSIST_SOC:
                ratio = max(ratio, 1.0)
        return int(np.argmin(np.abs(RATIOS - ratio)))

    return guarded
