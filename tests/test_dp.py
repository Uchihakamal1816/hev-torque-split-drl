import numpy as np
import torch
from stable_baselines3 import PPO
from torch import nn

from hev.controllers import rule_based_action
from hev.cycles import generate_cycle
from hev.dp import DPSolution, step_all_actions
from hev.env import HEVTorqueSplitEnv
from hev.evaluation import dp_factory, run_episode, static
from hev.imitation import as_policy, make_net
from hev.vehicle import G_FUEL_PER_KWH


def test_vectorised_step_matches_env_exactly():
    env = HEVTorqueSplitEnv()
    speed = generate_cycle("mixed", 300, np.random.default_rng(7))
    env.reset(options={"speed": speed, "soc0": 0.6})
    rng = np.random.default_rng(0)
    done, t = False, 0
    while not done:
        a = int(rng.integers(6))
        soc, eprev = env._soc, env._engine_on_prev
        soc_next, _, cost = step_all_actions(env._t_req[t], env._omega[t], np.array([soc]), np.array([eprev]), env.p, 1.0, env.start_cost)
        _, r, _, done, _ = env.step(a)
        env_cost = -r * env.reward_scale
        if done:
            env_cost -= (env._soc0 - env._soc) * env.p.nominal_energy_kwh * G_FUEL_PER_KWH
        assert abs(env_cost - cost[a, 0]) < 1e-9
        assert abs(soc_next[a, 0] - env._soc) < 1e-12
        t += 1


def test_dp_beats_rule_based_on_its_own_objective():
    env = HEVTorqueSplitEnv()
    speed = generate_cycle("urban", 600, np.random.default_rng(3))
    dp = run_episode(env, dp_factory, speed)
    rb = run_episode(env, static(rule_based_action), speed)
    assert dp["objective_l_eq_100km"] <= rb["objective_l_eq_100km"]


def test_dp_best_action_is_valid():
    sol = DPSolution(generate_cycle("highway", 120, np.random.default_rng(1)))
    assert 0 <= sol.best_action(10, 0.6, False) < 6


def test_ppo_warm_start_reproduces_imitation_policy(tmp_path):
    from finetune_ppo import load_warm_start

    torch.manual_seed(0)
    net = make_net()
    path = tmp_path / "il.pt"
    torch.save(net.state_dict(), path)
    env = HEVTorqueSplitEnv(cycle_duration_s=50)
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": nn.Tanh}, device="cpu")
    load_warm_start(model, path)
    il = as_policy(net)
    obs = np.random.default_rng(0).normal(size=(200, 6)).astype(np.float32)
    assert all(il(o) == int(model.predict(o, deterministic=True)[0]) for o in obs)
