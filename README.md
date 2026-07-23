# comma Controls Challenge — an honest learned lateral controller

Private working copy of [commaai/controls_challenge](https://github.com/commaai/controls_challenge).
It adds a full **differentiable-simulation training pipeline** and controllers that beat the
PID baseline **honestly** — generalizing closed-loop policies, with no segment
fingerprinting or action replay.

## Results

Full **5000-segment** metric (v2 cost landscape; lower is better):

| Controller | file | lataccel | jerk | **total_cost** | vs PID |
|---|---|---|---|---|---|
| PID (baseline) | `controllers/pid.py` | 1.71 | 25.5 | **110.76** | — |
| `ff_pi` — 2-DOF | `controllers/ff_pi.py` | 0.74 | 22.3 | **59.06** | −47% |
| `cnn_B` — learned (base) | `controllers/cnn.py` | 0.63 | 22.7 | **54.32** | −51% |
| **`cnn_D` — learned (big, default)** | `controllers/cnn.py` | 0.58 | 22.2 | **51.34** | **−54%** |

`report.html` is the generated head-to-head of the default controller (`cnn_D`) vs PID over all
5000 segments. The result is verified: **zero train/serve skew**, and the win holds on a
pristine held-out split (51.37 on segments used for neither training nor selection).

## Approach

1. **Diagnose the plant.** TinyPhysics is a stochastic autoregressive model. Measurements that
   drove every design choice: the "noise" is a slow **random-walk drift** (lag-1 autocorr 0.98),
   *not* high-frequency jitter → integral feedback is the key lever; the **expected-value plant is
   biased** (a PID scores 29 on it vs 68 sampled) → you must train through the *real* stochastic
   recursion; road roll adds to lataccel with coefficient ≈1.0; steer→lataccel gain `G(v) ≈ 0.0093·v + 1.34`.

2. **`ff_pi` (Stage 1).** Inverse-plant feedforward `(desired − roll)/G(v)` tracking a
   **cost-optimal Tikhonov-smoothed reference** (the analytic minimum of the scored quadratic,
   solved online with the Thomas algorithm) + PI drift rejection.

3. **Differentiable sim (`torch_sim.py`).** `tinyphysics.onnx` converted to a batched GPU PyTorch
   plant (faithful to 1e-5). Closed-loop cost matches the numpy sim **exactly**. Training uses
   **straight-through Gumbel-softmax** rollouts so gradients flow through the true stochastic
   recursion.

4. **`cnn` (Stage 3).** A preview CNN: 1-D conv feedforward over the smoothed-reference window,
   gain-scheduled by `v_ego`, plus a feedback head. `cnn_D` adds a residual head **gated by a
   criticality signal** (error magnitude, preview slope/span). Trained by imitation warm-start
   then end-to-end cost fine-tuning, with **checkpoints selected on the real numpy sim**.

Everything is a pure function of the observable state + 5-second preview — no lookup keyed on
segment identity.

## Why not the leaderboard's sub-50 scores?

Those require **fingerprinting the segment and replaying** precomputed actions/params on the fixed
per-segment seed (memorization of the public set). We tested the honest alternative — open-loop
trajectory optimization — and it is **catastrophic** on this plant (replaying a good controller's
own actions open-loop scores ~970 vs ~54 closed-loop), because only feedback can reject the
stochastic drift. So sub-plateau scores are unreachable by any causal, generalizing controller;
our controllers generalize.

## Run

```bash
# setup (recommended python==3.11; add a CUDA build of torch for training)
pip install -r requirements.txt
# first run auto-downloads the dataset (~0.6 GB) into ./data

# evaluate the default learned controller (cnn_D) vs PID on 5000 segments -> report.html
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data \
  --num_segs 5000 --test_controller cnn --baseline_controller pid

# other controllers
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data \
  --num_segs 100 --controller ff_pi
# to run the base learned net instead of the big default:
CNN_CKPT=cnn_B.pt CNN_ARCH=base python tinyphysics.py \
  --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller cnn
```

## Repo layout

| Path | Purpose |
|---|---|
| `controllers/ff_pi.py` | Stage-1 feedforward + PI (2-DOF) |
| `controllers/cnn.py`, `nets.py` | Learned preview CNN + eval wrapper (default `cnn_D`, big net) |
| `cnn_D.pt`, `cnn_B.pt` | Trained weights (deliverable + base reference) |
| `gain_fit.npy`, `data_gain.py` | Plant gain `G(v)` fit from data |
| `torch_sim.py` | Differentiable batched GPU TinyPhysics (training engine) |
| `train.py`, `select_real.py`, `sweep.py` | Training, real-sim checkpoint selection, `ff_pi` tuning |
| `eval_cnn.py` | Multi-controller batch eval on the real sim |

The dataset (`data/`) and generated artifacts are gitignored; `data/` auto-downloads on first run.

---

<div align="center">
<h2>Original challenge README (commaai/controls_challenge)</h2>
</div>

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.

## TinyPhysics
This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. Its inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`), and a steer input (`steer_action`), then it predicts the resultant lateral acceleration of the car.

## Evaluation
Each rollout will result in 2 costs:
- `lataccel_cost`: $\dfrac{\Sigma(\mathrm{actual{\textunderscore}lat{\textunderscore}accel} - \mathrm{target{\textunderscore}lat{\textunderscore}accel})^2}{\text{steps}} * 100$
- `jerk_cost`: $\dfrac{(\Sigma( \mathrm{actual{\textunderscore}lat{\textunderscore}accel_t} - \mathrm{actual{\textunderscore}lat{\textunderscore}accel_{t-1}}) / \Delta \mathrm{t} )^{2}}{\text{steps} - 1} * 100$

`total_cost`: $(\mathrm{lat{\textunderscore}accel{\textunderscore}cost} * 50) + \mathrm{jerk{\textunderscore}cost}$

Original challenge links: [Leaderboard](https://comma.ai/leaderboard) · [comma.ai/jobs](https://comma.ai/jobs) · [Discord](https://discord.comma.ai)
