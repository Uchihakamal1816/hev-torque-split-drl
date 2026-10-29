# Intelligent HEV Torque-Split Controller

A learned energy-management controller for a parallel hybrid electric vehicle. Every second it
decides how much of the traction torque the engine supplies and how much the electric motor
supplies, to minimise fuel use and battery wear.

**Method:** a dynamic-programming (DP) expert computes the optimal torque split offline on
training drive cycles. A neural network learns to copy it (DAgger imitation learning) using only
the 6 sensor signals a real controller has. A small SOC safety guard keeps the battery inside its
healthy window. The result runs causally, needs no future knowledge, and is evaluated on cycles it
never saw.

## Headline result

Synthetic traffic test set: 40 unseen cycles, mean over cycles, ± std across 3 training seeds.

| Controller | Full cost (L-eq/100 km) | Fuel + degradation (L-eq/100 km) | Fuel (L/100 km) | Battery degradation (g-eq) | Engine starts |
|:--|--:|--:|--:|--:|--:|
| DP, optimal but needs the future (bound) | 4.648 | 4.518 | 3.708 | 79.7 | 10.2 |
| **Ours: imitation of DP + SOC guard** | **4.845 ± 0.010** | **4.566 ± 0.002** | **3.701 ± 0.001** | **85.4** | **9.9** |
| Imitation of DP, no guard | 4.864 ± 0.007 | 4.560 ± 0.003 | 3.700 ± 0.001 | 84.4 | 9.9 |
| Imitation + PPO fine-tune | 4.821 ± 0.045 | 4.614 ± 0.014 | 3.698 ± 0.002 | 89.5 | 8.1 |
| Rule-based (thermostat with hysteresis) | 5.464 | 5.096 | 3.773 | 150.1 | 41.2 |
| DQN trained from scratch | 6.241 | 5.475 | 3.911 | 191.0 | 37.6 |
| ICE only | 9.285 | 4.689 | 3.989 | 71.5 | 65.3 |

Against the rule-based controller, ours is **11.3 % lower on full cost, 10.4 % lower on fuel +
degradation, 1.9 % lower on fuel, with 43 % less battery degradation and 76 % fewer engine
starts**, and it comes within about 4 % of the DP optimum on full cost.

- **Full cost** is exactly what every controller is trained to minimise: fuel, battery
  degradation, engine starts, SOC-window violations and unmet torque, with the battery energy
  change converted to fuel. It is the fairest single score.
- **Fuel + degradation** is the project objective without the penalty terms.
- **Fuel** is SOC-corrected: any change in battery energy is converted to fuel at about 270 g/kWh.

### Real WLTC cycles (class 3a and 3b)

| Controller | Full cost | Fuel + degradation | Fuel (L/100 km) | Degradation (g-eq) | SOC outside window | Unmet torque (Nm) |
|:--|--:|--:|--:|--:|--:|--:|
| DP (bound) | 5.433 | 5.246 | 3.530 | 297.0 | 0.0 % | 0.0 |
| **Ours: imitation of DP + SOC guard** | **6.228 ± 0.287** | **5.494 ± 0.024** | 3.571 ± 0.002 | **332.9** | 1.0 % | 6.3 |
| Imitation of DP, no guard | 11.860 ± 2.844 | 5.379 ± 0.057 | 3.526 ± 0.004 | 320.7 | 18.3 % | 6.3 |
| Imitation + PPO fine-tune | 10.398 ± 2.871 | 5.387 ± 0.055 | 3.521 ± 0.006 | 322.9 | 9.7 % | 6.2 |
| Rule-based | 6.871 | 5.972 | **3.560** | 417.4 | 0.0 % | 20.5 |
| DQN from scratch | 6.598 | 6.283 | 3.789 | 431.6 | 0.0 % | 0.0 |
| ICE only | 123.579 | 4.732 | 4.206 | 91.0 | 61.8 % | 0.0 |

**Read this table with care:**

- **The SOC guard was added after the first WLTC evaluation.** Without it, the imitation policy kept
  the battery above the 0.75 SOC ceiling 18 % of the time on WLTC: WLTC reaches 131 km/h, faster
  than any training cycle. The guard's thresholds come from the SOC window, not from tuning, and it
  was checked on the validation split before re-running the test. Still, the WLTC numbers for the
  guarded controller are **not a clean held-out result**. A clean confirmation needs new test
  cycles (see next steps).
- On WLTC, ours wins on full cost (−9.4 %), fuel + degradation (−8.0 %) and degradation (−20 %), and
  delivers more of the requested torque (6.3 vs 20.5 Nm unmet). It is **0.3 % worse on fuel alone**
  (3.571 vs 3.560 L/100 km), the price of the guard.
- ICE only looks good on fuel + degradation only because it parks the battery at its 0.80 ceiling
  (62 % of the time outside the window). Its full cost shows the real picture.

## Why it works, and what did not

- **Fuel headroom is small, battery headroom is large.** Even the DP optimum, which knows the
  whole future, is only about 1.5 % better than rule-based on fuel alone. Most of the gain is
  using the battery less harshly: 43 % less degradation and 76 % fewer engine starts.
- **The teacher matters.** A DQN trained from scratch with the same reward lost to the rule-based
  controller on every test set. Imitating the DP expert gets within 1 % of the optimum on
  validation.
- **RL fine-tuning did not help.** PPO, initialised from the imitation weights, cut engine starts
  further but scored worse on the validation objective, so it was not selected.

## Evaluation protocol

| Split | Cycles | Used for |
|:--|:--|:--|
| Train | UDDS, HWFET, synthetic seeds 5000–5014 per pattern (62 cycles) | DP labels, DAgger, PPO, DQN |
| Validation | synthetic seeds 2000–2009 per pattern (40 cycles) | every method and hyperparameter choice |
| Test | synthetic seeds 1000–1009 per pattern (40 cycles), WLTC 3a and 3b | final report |

Validation choices: engine-start cost 1 g over 3 g, and imitation without PPO over imitation with
PPO, both by the fuel + degradation objective. Three training seeds per learned method.

## A bug found along the way

The first test run showed ICE only ending a WLTC cycle at 100 % SOC, which the model should forbid.
`allocate()` clipped the motor torque to its limits and then overwrote it with the unclipped value,
so braking energy beyond the 20 kW charge limit, or above the SOC ceiling, still charged the battery
for free. It is fixed in `hev/vehicle.py` and in the DP mirror, with regression tests. **Every model
was retrained and every number in this README comes from the corrected simulator.** Results
published in earlier commits are invalid.

## Model

`hev/vehicle.py`: a quasi-static parallel (P2-style) powertrain, roughly Prius-class: 1400 kg,
115 Nm engine with a BSFC map, 160 Nm motor at 92 % efficiency, 6.5 Ah / 1.31 kWh battery with
27 kW discharge and 20 kW charge limits, SOC hard limits 0.30–0.80, and a throughput-based
degradation proxy (20 g-eq per Ah, amplified above 40 A).

Action: engine share of the traction torque, one of 0, 0.25, 0.5, 0.75, 1.0, 1.25 (above 1
charges the battery). Observation: SOC, speed, acceleration, torque request, 5 s power preview,
previous engine state.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests                                   # 29 tests
python scripts/prepare_real_cycles.py                       # raw files -> data/cycles/*.csv
python train_imitation.py --seed 0 --start-cost 1 --out models/il/il_sc1_seed0.pt   # about 8 min on CPU
python finetune_ppo.py --init models/il/il_sc1_seed0.pt --out models/ppo/ilppo_sc1_seed0
python train.py --timesteps 300000 --real-train --out models/dqn_real               # DQN baseline
python validate.py                                          # validation split, for choices
python evaluate.py                                          # test split, final report
```

## Layout

```
hev/vehicle.py      powertrain, battery, fuel and degradation model
hev/cycles.py       synthetic traffic patterns
hev/env.py          Gymnasium environment (actions, reward, summaries)
hev/dp.py           dynamic-programming optimal controller (bound and teacher)
hev/imitation.py    DAgger imitation of the DP expert
hev/controllers.py  rule-based and ICE-only baselines, SOC safety guard
hev/datasets.py     real drive-cycle loader
hev/evaluation.py   fixed train / validation / test splits and episode runner
data/               EPA and WLTC cycle files and their sources
train_imitation.py  train the imitation controller
finetune_ppo.py     PPO fine-tuning from imitation weights
train.py            DQN baseline
validate.py         score candidates on validation
evaluate.py         final test report and plot
results/            test and validation tables, report, plot, logs
models/             trained models (il/, ppo/, dqn_real.zip)
```

## Limitations

- Quasi-static model, single fixed gear, no thermal model, no road grade.
- Parameters are plausible but uncalibrated. Compare controllers with each other; do not quote
  the absolute L/100 km as a real vehicle figure.
- Degradation is a throughput proxy, not an electrochemical ageing model.
- WLTC with the guard is not a clean held-out result (see above), and only two real test cycles exist.

## Roadmap after half-term

1. **Prove it on new roads.** Clean held-out tests on EPA US06, SC03 and NYCC, and the Indian MIDC cycle, none of which played any part in a decision.
2. **Raise model fidelity.** Cross-check the vehicle model against NREL FASTSim; add a multi-speed gearbox, road grade and a calibrated battery ageing model.
3. **Smarter controller.** Add high-speed segments to training traffic so the policy keeps SOC in its window without the guard; add route and traffic preview.
4. **Toward a real ECU.** Export the network to ONNX, measure its latency on embedded hardware, run hardware-in-the-loop tests, and write the final report.
