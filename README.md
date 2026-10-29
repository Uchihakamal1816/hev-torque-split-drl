# Deep RL Torque-Split Optimisation for a Parallel Hybrid Electric Vehicle

Rule-based hybrid controllers (thermostat / state-machine logic) are replaced by a
learned policy. A DQN agent chooses, every second, how much of the traction torque
the internal combustion engine (ICE) supplies versus the electric motor (EM). It is
trained on randomised traffic (stop-and-go urban, suburban, highway, and mixed) to
minimise fuel consumption plus a battery-degradation proxy.

## Results

All numbers are mean L/100 km (SOC-corrected fuel, lower is better) on **held-out** cycles that the
policy was not trained on.

**Synthetic traffic** (4 patterns × 10 unseen seeds, 40 cycles):

| Policy | L/100 km | ΔSOC | Engine starts / cycle | Engine on-time | Degradation proxy (g-eq) |
|:--|--:|--:|--:|--:|--:|
| **Rule-based** | **3.73** | −0.11 | 40.0 | 50.4 % | 262.0 |
| DQN, trained on real + synthetic | 3.77 | −0.15 | 31.4 | 55.3 % | 249.9 |
| DQN, trained on synthetic only | 3.80 | −0.14 | 25.1 | 57.6 % | 291.2 |
| ICE only | 3.94 | +0.12 | 65.3 | 82.5 % | 168.9 |

**Real published cycles** (WLTC class 3a and 3b, never used in training):

| Policy | L/100 km | ΔSOC | Engine starts / cycle | Unmet torque (Nm) |
|:--|--:|--:|--:|--:|
| **Rule-based** | **3.54** | −0.03 | 53.0 | 20.5 |
| DQN, trained on synthetic only | 3.66 | −0.11 | 81.0 | 13.6 |
| DQN, trained on real + synthetic | 3.69 | −0.15 | 60.5 | 0.0 |
| ICE only | 3.80 | +0.40 | 61.5 | 0.0 |

Per-pattern tables are in `results/summary_by_pattern.csv`, per-cycle rows in `results/per_cycle.csv`
and `results/real_cycles.csv`. The WLTC class 3b trace is in `results/wltc3b_soc_fuel.png`.

**Honest reading**

- Both DQN policies beat the ICE-only baseline on every held-out set, and the real-data DQN has no unmet torque.
- Neither DQN beats the rule-based controller. The gap is smallest on synthetic traffic (3.77 vs 3.73) and largest on WLTC (3.69 vs 3.54).
- Training on UDDS and HWFET improved the synthetic result slightly (3.80 → 3.77) and made the WLTC result no better (3.66 → 3.69). Real-data training is not a demonstrated win.
- The rule-based controller's WLTC result is partly flattered: it leaves 20.5 Nm of traction torque unmet in total, which the fuel metric does not count. The DQN trained on real data delivers the torque in full.
- Each evaluation uses one training seed, so differences of about 0.03 L/100 km are within what another seed could change.

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
python -m pytest -q tests                                        # 20 tests
python scripts/prepare_real_cycles.py                            # raw files -> data/cycles/*.csv
python train.py --timesteps 300000 --out models/dqn_synthetic    # synthetic traffic only
python train.py --timesteps 300000 --real-train --out models/dqn_real  # + UDDS and HWFET
python evaluate.py                                               # held-out tables and plots
```

Training takes about 5 to 10 minutes per model on CPU.

## Layout

```
hev/vehicle.py      powertrain, battery, fuel and degradation model
hev/cycles.py       synthetic traffic patterns
hev/env.py          Gymnasium environment (RATIOS, reward, summaries)
hev/controllers.py  rule-based and ICE-only baselines
hev/datasets.py    loader for the real drive cycles
data/              raw EPA and WLTC files, converted cycles, sources
scripts/           converts the raw files to speed traces
train.py           DQN training (Stable-Baselines3)
evaluate.py        held-out comparison, tables and plots
tests/             pytest suite (20 tests)
models/            trained policies
results/           evaluation output and training logs
```

## Limitations

- Quasi-static, single fixed gear, no thermal model, no grade, no engine transients.
- Parameters are plausible but uncalibrated. Absolute L/100 km should not be quoted as a
  real vehicle figure; the relative comparison is the point.
- Degradation is a proxy, not a validated ageing model.
- Single training seed per model. Results would need multiple seeds before any strong claim.
- Only two real cycles are used for training and two for testing, so real-data conclusions are narrow.

## Next steps

1. Multi-seed training with a validation split separate from the test cycles.
2. Penalise engine starts more heavily, or add a minimum on-time, to fix chattering.
3. Warm-start from the rule-based controller (imitation learning), or try PPO with a continuous torque split.
4. Replace the degradation proxy with a calibrated ageing model.
