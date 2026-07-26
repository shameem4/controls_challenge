# comma Controls Challenge — an honest learned lateral controller

Private working copy of [commaai/controls_challenge](https://github.com/commaai/controls_challenge).
It adds a **differentiable-simulation training pipeline** and controllers that beat the PID
baseline **honestly** — generalizing closed-loop policies, with no segment fingerprinting or
action replay.

## Results

Full **5000-segment** metric (v2 cost landscape; lower is better). This is the same cost landscape
the official leaderboard uses — its PID baseline is 110.25 and ours measures 110.76 (see caveat 2
below on how precisely those are comparable).

| Controller | file | lataccel | jerk | **total_cost** | vs PID |
|---|---|---|---|---|---|
| PID (baseline) | `controllers/pid.py` | 1.71 | 25.51 | **110.76** | — |
| `pid_w_ff` — ported reference (jonoomph) | `controllers/pid_w_ff.py` | 0.71 | 23.93 | **59.49** | −46% |
| `ff_pi` — 2-DOF feedforward + PI | `controllers/ff_pi.py` | 0.74 | 22.33 | **59.06** | −47% |
| `ff_pi_tuned` — same, CMA-ES tuned | `controllers/ff_pi_tuned.py` | — | — | **54.56** | −51% |
| `ff_pi_rl2` — + rate-limit anti-windup | `controllers/ff_pi_rl2.py` | — | — | **52.30** | −53% |
| **`cnn` — learned preview net (default)** | `controllers/cnn.py` | **0.545** | **20.61** | **47.87** | **−57%** |

`ff_pi_tuned` re-tunes the six `ff_pi` parameters with CMA-ES on a 400-segment set disjoint from
every eval split (component costs not recorded for the 5000 run, hence the dashes). Tuning on only
60 segments produced a 16% *apparent* gain that was almost entirely overfitting — this metric's
subset noise is large enough that small tuning sets fit the sample, not the controller.

The default `cnn` controller scores **47.87** on the full 5000 and **48.14** on a pristine
held-out split it never saw for training *or* checkpoint selection. It is a pure function of the
observed state, the 5-second preview, and its own recent actions — **no per-segment memorization**.

Two independently-designed 2-DOF controllers (ours and a ported reference) land 0.5 apart at ~59 —
indistinguishable at this metric's noise level — which is what pins the **classical
feedforward+feedback plateau** on this cost landscape. The learned net clears it by ~11 points,
a margin well outside the noise.

**Where that margin actually comes from.** It is not broad superiority, and the earlier wording
here implied otherwise. On the clean 500-segment split `ff_pi_tuned` scores 55.68 and `cnn` 48.14;
tracing every segment individually shows that **six segments out of 500 account for 72% of that
7.54-point gap** — `cnn` averages 142 on those six against the classical controller's 596. They are
low-speed, large-lateral-acceleration corners, and the classical controller's loss there arrives in
short bursts (71% of its squared error in 10% of the timesteps). The learned net's demonstrated
advantage is that **it does not blow up on these segments**, not that it tracks better everywhere.

**The mechanism, measured.** The plant clamps its own lataccel change at `MAX_ACC_DELTA = 0.5` per
step. That clamp sits ~11× above normal operation (a typical jerk cost of ~20 implies RMS lataccel
change ~0.045/step), and since jerk is charged quadratically at 10000×, **one saturated step costs
25–64× a normal step**. On the worst segments a handful of timesteps — sometimes three — produce
27–62% of the entire jerk cost. Comparing controllers over the eight worst segments makes the causal
role plain: `ff_pi_tuned` accumulates **97** saturated steps against `cnn`'s **2**, and the
per-segment correspondence is near-exact — where `ff_pi` saturates, `cnn` removes it and cost falls
4–8×; where saturation is already zero, `cnn` gains 0–1.2×, and on the one segment with no saturated
steps at all it does not help (282.3 vs 284.1). Trajectory optimisation through the differentiable
plant confirms this is self-inflicted rather than intrinsic: the optimal trajectory is rate-saturated
**0.00%** of the time on five of those six segments.

That is what `ff_pi_rl2` fixes, with **conditional** anti-windup — freeze the PI integrator while the
clamp binds, held for the plant's measured 3-step dead time. `i_clip` does not address it, being a
*fixed* magnitude clamp already at its optimum. Across all 20,000 segments only 626 (3.1%) ever
saturate, but they cost 6.4× a clean segment and hold **17.2% of all cost**. Segments where the
mechanism never fires are bit-identical to `ff_pi_tuned`, so this costs nothing on the other 96.9%.
The −2.26 gain replicates on 15,000 segments never evaluated elsewhere here (−1.89, bootstrap 95% CI
[−2.35, −1.48]); the claim rests on that CI and a distribution-free sign test (389 improved / 231
worsened of 620 activated, p=2.3e-10) rather than a t-test, since the paired deltas are heavy-tailed.

Note `data/SYNTHETIC` contains **20,000** segments; the "full 5000" metric quoted here is `ALL[:5000]`.
On all 20,000, `ff_pi_tuned` scores 54.88 and `ff_pi_rl2` 52.90.

Verified by an adversarial review of the result:

- **zero train/serve skew** — the numpy eval path reproduces the torch training path to 1e-7;
- **no memorization** — the controller is a pure function of its arguments (identical outputs from
  fresh and sequential instances) and scores are invariant to segment evaluation order, so nothing
  leaks across segments; it reads no file at eval beyond its own weights;
- **the score comes from learning** — the same architecture with random weights scores 1329 where
  the trained net scores 61 (a zero-steer controller scores 1327);
- **numerically clean** — no NaN/inf, 0% action saturation, and finite output for every
  `future_plan` length from 50 down to empty;
- every improvement was confirmed on segments disjoint from training *and* checkpoint selection.

`report.html` is the generated head-to-head vs PID over all 5000 segments.

Two caveats stated plainly:

**1. The headline metric includes trained-on segments** (2000 of the 5000 here), so **48.14 on the
clean split is the honest measure of quality**; 47.87 is the comparable-to-others number.

**2. This metric is strongly subset-dependent, so treat small cross-entry gaps as noise.** Measured
on two disjoint 1000-segment subsets:

| Subset | PID | `cnn` | ratio |
|---|---|---|---|
| `[0:1000]` (inside the headline range) | 107.01 | 47.20 | 0.441 |
| `[5000:6000]` (never trained *or* selected on) | 114.64 | 49.79 | **0.434** |

PID alone swings 7.6 points between subsets, so absolute scores carry several points of
uncertainty and our 110.76 baseline vs the published 110.254 does not by itself prove an identical
evaluation set. The **ratio to PID is the robust statistic** — and it is essentially unchanged
(0.441 → 0.434) on 1000 segments the controller never saw, which is the real evidence that it
generalizes rather than memorizes.

## Where this sits on the leaderboard

| Score band | What lives there |
|---|---|
| **~7–30** | **Exploits.** Per-segment action/parameter optimization replayed via a segment fingerprint, and "online sim probing with RNG reset" — the leaderboard's own entry descriptions say so. These memorize the public set's fixed per-segment seeds. |
| **~36–49** | **A mix.** Genuine honest controllers — MPC on a linear LPV-ARX model (~36), PPO policies (~42–46), tube-MPC and custom feedback controllers (~48–49) — *plus* several more per-segment exploits. |
| **~50–60** | Honest classical controllers: PID+FF, 2-DOF, evolution-tuned feedback. |

Our 47.87 is an honest, generalizing controller that sits **mid-pack among the honest entries**: it
clears the classical plateau by ~11 points and is at or slightly ahead of the well-known ML_PID
entry (50.63) — though per caveat 2 above, a few points is within subset uncertainty, so treat that
particular comparison as a tie rather than a win. A handful of MPC- and PPO-based controllers score
clearly lower. **The honest frontier is ~36**, not ~50 — reaching it is a method change (see below),
not a tuning gap.

Why the exploits need the fixed seed: we tested the honest version of their idea — optimize an
open-loop action sequence offline, then run it. Replaying even a *good* controller's own actions
open-loop scores **~970** versus **~54** closed-loop, because only feedback can counteract the
plant's stochastic drift. Per-segment optimized actions only work when replayed against the exact
noise realization they were tuned for.

## Approach

1. **Diagnose the plant.** TinyPhysics is a stochastic autoregressive model, and the measurements
   drove every later decision:
   - the "noise" is a slow **random-walk drift** (lag-1 autocorrelation 0.98, only ~4% of energy
     above 0.7 Hz) — *not* high-frequency jitter, so **integral feedback** is the key lever and
     low-pass filtering is useless;
   - the **expected-value plant is biased** (a PID scores 29 on it but 68 sampled), so training and
     model selection must run through the *real stochastic* recursion;
   - road roll adds to lataccel with coefficient ≈1.0; the steer→lataccel gain fits
     `G(v) ≈ 0.0093·v + 1.34`.

2. **`ff_pi` — the classical baseline.** Inverse-plant feedforward `(desired − roll)/G(v)` tracking
   a **cost-optimal Tikhonov-smoothed reference** — the analytic minimum of the scored quadratic,
   solved online with the Thomas algorithm — plus PI drift rejection. The smoothing insight is
   lifted from what the top exploits compute offline, but applied causally over the preview window.

3. **Differentiable sim (`torch_sim.py`).** `tinyphysics.onnx` converted to a batched GPU PyTorch
   plant (logits faithful to 1e-5; closed-loop cost matches the numpy sim **exactly**). Plant modes:
   `expected`, `sample`, and **straight-through Gumbel-softmax** so gradients flow through the true
   stochastic recursion.

4. **`cnn` — the learned preview net** (`nets.py: AblNet`, cfg `PM`). Structure mirrors the
   classical design rather than replacing it with a black box:
   - 1-D **conv feedforward** over the (target − roll) preview window — a learned preview/FIR;
   - **FiLM gain-schedule** on `v_ego`;
   - a **feedback head**, plus a residual head **gated by a criticality signal** (error magnitude,
     preview slope/span);
   - two extra inputs isolated by ablation: **its own previous actions** and **multi-horizon preview
     errors** (current lataccel vs future-mean at near/mid/far).

5. **Training.** End-to-end on the exact challenge cost via Gumbel rollouts, with **soft-token full
   BPTT**: the plant feeds its lataccel back as a *discrete token*, which blocks gradients, so the
   embedding lookup is replaced by a straight-through **soft one-hot** (exact forward, differentiable
   backward). Gradients then flow through the full autoregressive recursion instead of a myopic
   1-step window — a real gain, mostly via lower jerk (+1.5 on the 5000 metric, +0.3 on a fully
   clean split). Trained on 2000 segments with a 60-step truncated-BPTT window; **checkpoints
   selected on the real numpy sim**, never the surrogate.

## What didn't work (and why)

Documented because the negative results were more informative than most of the wins.

| Tried | Outcome | Why |
|---|---|---|
| **Online MPC on the neural plant** (gradient and MPPI sampling) | Gradient diverged (9353); every sampling config landed 944–16237 — worse than steering zero (682) | The plant is **chaotic**: tiny action perturbations produce divergent predicted trajectories, so the expected-cost *ranking* of candidate action sequences is noise. Re-tested with the correct soft-token gradient — it converges better but still nowhere near usable. It's a **bad-landscape** problem, not a bad-gradient one. This is why the frontier's MPC runs on a smooth *linear* model. |
| **Shooting-teacher → distillation** | Open-loop replay of optimized actions: ~970 sampled | Open-loop cannot reject drift; only useful if you replay the exact seed (i.e. the exploit). |
| **Deterministic-only training** | 56.38 | Great *nominal* controller (jerk 19.1, the lowest we measured) but never learns drift rejection, and it overfits the noise-free plant if trained long. Its deterministic-plant optimum (~36) equals the honest MPC frontier — the noise-free ceiling. |
| **Deterministic base → noisy fine-tune** (two-stage) | 49.70 | Valid and marginally better than noisy-from-scratch (49.96) at the time; superseded by soft-token BPTT. |
| **Rate-limited delta actions** (`action = prev + tanh(·)·scale`) | Diverged | The pure integrator winds up under our diffsim training; needs PPO-style stabilisation. |
| **Curvature + rate features** (`lataccel/v²`, derivatives) | Hurt (~1.3) | Redundant with the multi-horizon error features, and dilutes a small net. |
| **Past-history temporal branch** | Hurt | The closed-loop diffsim training already captures the dynamics. |
| **K-sample gradient averaging** | Hurt substantially | Shrinks effective exploration per step at matched budget. |
| **Denoising the feedback** | Not applicable | This is **process** noise on a **fully-observed** state — the sampled lataccel *is* the car's real position and is what gets scored. There is no clean signal hiding underneath to recover. |

The remaining gap to the ~36 frontier is a **method** difference — MPC on a smooth linear plant
model, or value-based RL (PPO) — not another lever on this approach.

## Run

```bash
# setup (recommended python==3.11; add a CUDA build of torch for training)
pip install -r requirements.txt
# first run auto-downloads the dataset (~0.6 GB) into ./data

# evaluate the default learned controller vs PID over all 5000 segments -> report.html
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data \
  --num_segs 5000 --test_controller cnn --baseline_controller pid

# any other controller (pid, zero, ff_pi, pid_w_ff, cnn)
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data \
  --num_segs 100 --controller ff_pi

# retrain the deliverable: soft-token BPTT, 60-step window, 2000 segments
TBPTT=60 TRAIN_N=2000 python ablate.py PM 1000 gumbel_soft
python select_abl.py PM 240            # pick the best checkpoint on the real sim
```

`ablate.py <cfg> <iters> [gumbel|expected][_soft] [warmstart.pt]` — `cfg` toggles the input
features (`P` previous actions, `M` multi-horizon errors, `H` history branch), `_soft` enables
soft-token BPTT, and `TBPTT`/`TRAIN_N` set the BPTT window and training-set size.

## Repo layout

| Path | Purpose |
|---|---|
| `controllers/cnn.py`, `nets.py` | **Deliverable** learned preview net (`AblNet`, cfg `PM`) + eval wrapper |
| `cnn_PM.pt` | Trained weights for the default `cnn` controller |
| `controllers/ff_pi.py` | 2-DOF feedforward + PI baseline |
| `controllers/ff_pi_tuned.py`, `controllers/pid_tuned.py` | CMA-ES-tuned variants; parameterised copies so the quoted baselines stay untouched |
| `controllers/ff_pi_rl2.py` | Best classical controller (52.30) — adds conditional anti-windup against the plant's lataccel rate clamp |
| `tune_cma.py` | CMA-ES tuner (400-segment tune set, disjoint held-out guard) |
| `controllers/pid_w_ff.py` | Ported reference controller (jonoomph, attributed) — 59.49 on our 5000 |
| `torch_sim.py` | Differentiable batched GPU TinyPhysics (the training engine) |
| `ablate.py`, `select_abl.py` | Config/ablation training; real-sim checkpoint selection |
| `train.py`, `sweep.py`, `data_gain.py`, `gain_fit.npy` | Earlier training pipeline; `ff_pi` tuning; plant-gain fit |
| `eval_cnn.py` | Multi-controller batch eval on the real sim |

The dataset (`data/`), checkpoints (`ckpts/`) and `__pycache__` are gitignored; `data/`
auto-downloads on first run.

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
