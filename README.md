# Deep RL Torque-Split Optimisation for a Parallel Hybrid Electric Vehicle

Rule-based hybrid controllers (thermostat / state-machine logic) are replaced by a
learned policy. A DQN agent chooses, every second, how much of the traction torque
the internal combustion engine (ICE) supplies versus the electric motor (EM). It is
trained on randomised traffic (stop-and-go urban, suburban, highway, and mixed) to
minimise fuel consumption plus a battery-degradation proxy.

## Results (held-out cycles, 40 episodes: 4 traffic patterns x 10 unseen seeds)

| Policy     | L/100 km (SOC-corrected) | ΔSOC | Engine starts / cycle | Engine on-time | Degradation proxy (g-eq) |
|:-----------|-------:|-------:|-------:|------:|-------:|
| ICE only   | 3.94 | +0.12 | 65.3 | 82.5 % | 168.9 |
| Rule-based | **3.73** | -0.11 | 40.0 | 50.4 % | 262.0 |
| DQN        | 3.95 | -0.15 | 78.5 | 75.3 % | 270.1 |

Per-pattern numbers are in `results/summary_by_pattern.csv`; per-cycle rows are in
`results/per_cycle.csv`. The mixed-cycle SOC and cumulative-fuel trace is in
`results/mixed_cycle_soc_fuel.png`.

**Honest reading:** with the current reward and a 200k-step, single-seed DQN, the
learned policy does **not** beat the hand-written rule-based controller. It is slightly
worse overall and chatters more (more engine starts). The rule-based controller wins
clearly in urban traffic (2.31 vs 2.79 L/100 km for DQN, 3.05 for ICE only). The gap
is most likely a training-budget and reward-shaping issue, not a sign that RL cannot
help here, but that is not yet demonstrated. Likely next steps are listed below.

## Model

`hev/vehicle.py`, a quasi-static parallel (P2-style) powertrain, roughly Prius-class:

- Chassis: 1400 kg, Cd 0.28, A 2.2 m², Crr 0.012, fixed final drive 4.1, drive efficiency 0.96.
- Engine: 115 Nm max, 500 rad/s, BSFC bowl around an efficient island (230 g/kWh).
  The engine can run from standstill with a slipping clutch, so its speed is floored at idle for fuel purposes.
- Motor/generator: 160 Nm, 92 % efficiency. Battery: 6.5 Ah, 1.31 kWh, 180–225 V
  open-circuit voltage, 27 kW discharge / 20 kW charge limits, SOC window 0.30–0.80.
- Degradation proxy: 20 g-eq per Ah throughput, amplified at high current (C-rate knee at 40 A).
  This is a throughput-based stand-in, not an electrochemical ageing model.

Allocation (`allocate`): the action sets the engine torque to a share of the traction
request (0, 0.25, 0.5, 0.75, 1.0 or 1.25). Values above 1 push surplus through the
generator to charge the battery. The motor balances the remainder. If the motor saturates,
the engine picks up the shortfall, so the chosen split is a request rather than a guarantee.

## RL formulation (`hev/env.py`)

- **Action:** discrete, 6 engine-share levels above.
- **Observation (6-dim):** SOC, speed, acceleration, traction torque request, 5-second
  look-ahead power, engine state at the previous step.
- **Reward (negative cost, per second):** fuel burnt + degradation proxy + 1 g-eq per
  engine start + SOC-window penalty (outside 0.35–0.75) + unmet-torque penalty. At the
  end of each episode the change in battery energy is converted to fuel-equivalent
  (about 270 g per kWh), so draining the battery is not free.
- **Traffic:** `hev/cycles.py` generates a new 1200 s speed trace per episode from a
  traffic pattern. Mixed traces switch regime every segment.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests          # 15 tests: physics, env contract, controllers
python train.py --timesteps 200000 # writes models/dqn_hev.zip (about 3 min on CPU)
python evaluate.py                 # writes results/ (tables and plot)
```

## Layout

```
hev/vehicle.py      powertrain, battery, fuel and degradation model
hev/cycles.py       synthetic traffic patterns
hev/env.py          Gymnasium environment (RATIOS, reward, summaries)
hev/controllers.py  rule-based and ICE-only baselines
train.py            DQN training (Stable-Baselines3)
evaluate.py         held-out comparison, tables and plot
tests/              pytest suite
models/             trained policy (dqn_hev.zip)
results/            evaluation output and training log
```

## Limitations

- Quasi-static, single fixed gear, no thermal model, no grade, no engine transients.
- Parameters are plausible but uncalibrated. Absolute L/100 km should not be quoted as a
  real vehicle figure; the relative comparison is the point.
- Degradation is a proxy, not a validated ageing model.
- Single training seed. Results would need multiple seeds before any strong claim.

## Next steps

1. Multi-seed training and a validation split separate from the test seeds.
2. Penalise engine starts more heavily, or add a minimum on-time, to fix chattering.
3. Try an approach that pins the rule-based controller as a warm start, or PPO with
   continuous torque split.
4. Replace the degradation proxy with a calibrated ageing model.
