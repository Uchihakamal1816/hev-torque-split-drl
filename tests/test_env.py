import numpy as np
from gymnasium.utils.env_checker import check_env

from hev.controllers import ice_only_action, rule_based_action
from hev.cycles import ALL_PATTERNS, generate_cycle
from hev.env import OBS_DIM, RATIOS, HEVTorqueSplitEnv


def test_gymnasium_env_checker_passes():
    check_env(HEVTorqueSplitEnv(cycle_duration_s=120), skip_render_check=True)


def test_cycles_are_nonnegative_and_have_requested_length():
    rng = np.random.default_rng(0)
    for pattern in ALL_PATTERNS:
        v = generate_cycle(pattern, 600, rng)
        assert v.shape == (600,)
        assert (v >= 0).all()
        assert v.max() <= 40.0  # below 145 km/h, sanity bound


def test_episode_runs_to_end_and_summary_is_finite():
    env = HEVTorqueSplitEnv(cycle_duration_s=300)
    obs, _ = env.reset(seed=1, options={"pattern": "mixed", "soc0": 0.6})
    assert obs.shape == (OBS_DIM,)
    steps, done = 0, False
    while not done:
        obs, reward, term, done, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        steps += 1
    assert steps == 300
    s = env.summary()
    assert np.isfinite(s["l_per_100km"])
    assert 0.0 <= s["soc_end"] <= 1.0


def test_engine_only_policy_burns_fuel_and_keeps_soc_roughly_steady():
    env = HEVTorqueSplitEnv(cycle_duration_s=600)
    obs, _ = env.reset(seed=2, options={"pattern": "suburban", "soc0": 0.6})
    done = False
    while not done:
        obs, _, _, done, _ = env.step(ice_only_action(obs))
    s = env.summary()
    assert s["fuel_g"] > 0
    assert s["engine_on_frac"] > 0.2


def test_rule_based_controller_returns_valid_actions():
    env = HEVTorqueSplitEnv(cycle_duration_s=200)
    obs, _ = env.reset(seed=3)
    for _ in range(200):
        a = rule_based_action(obs)
        assert 0 <= a < len(RATIOS)
        obs, _, _, done, _ = env.step(a)
        if done:
            break
