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
| `pid_boot` — PID + physics lookahead + smoothing + bootstrap | `controllers/pid_boot.py` | 0.94 | 21.50 | **68.41** | −38% |
| `pid_w_ff` — ported reference (jonoomph) | `controllers/pid_w_ff.py` | 0.71 | 23.93 | **59.49** | −46% |
| `ff_pi` — 2-DOF feedforward + PI | `controllers/ff_pi.py` | 0.74 | 22.33 | **59.06** | −47% |
| `ff_pi_tuned` — same, CMA-ES tuned | `controllers/ff_pi_tuned.py` | — | — | **54.56** | −51% |
| `ff_pi_rl2` — + rate-limit anti-windup | `controllers/ff_pi_rl2.py` | — | — | **52.30** | −53% |
| `ff_pi_boot` — + bootstrapped integrator | `controllers/ff_pi_boot.py` | 0.63 | 19.92 | **51.22** | −54% |
| `ff_pi_tau` — + preview-gated feedforward detune (best classical) | `controllers/ff_pi_tau.py` | 0.605 | 19.22 | **49.47** | −55% |
| **`cnn` — learned preview net (default, `cnn_v3.pt`)** | `controllers/cnn.py` | — | — | **45.74** | **−59%** |
| `cnn` with `cnn_v2.pt` — same recipe, stopped at 400 iters | `controllers/cnn.py` | — | — | **46.91** | −58% |
| `cnn` with `cnn_dual.pt` — BC→PO, tied with `cnn_v2` | `controllers/cnn.py` | 0.531 | 20.33 | **46.89** | −58% |
| `cnn` with `cnn_PM.pt` — v1 tag, a below-average run | `controllers/cnn.py` | 0.545 | 20.61 | **47.87** | −57% |

`ff_pi_tuned` re-tunes the six `ff_pi` parameters with CMA-ES on a 400-segment set disjoint from
every eval split (component costs not recorded for the 5000 run, hence the dashes). Tuning on only
60 segments produced a 16% *apparent* gain that was almost entirely overfitting — this metric's
subset noise is large enough that small tuning sets fit the sample, not the controller.

**`cnn_v3.pt` (46.91 → 45.74) — the earlier checkpoints were undertrained.** Identical architecture
and recipe to `cnn_v2`, trained 6.75x longer (2700 iterations / 10,800 rollouts against 400 / 1,600).
It beats `cnn_v2` by −1.167 [−1.91, −0.64] on the headline 5000 with median −0.364 and 3016/5000
segments improved (sign z=+14.6), and it is the first cnn result here to improve mean, median, p99
and win-rate together rather than trading median for tail. The ~48 ceiling that eleven interventions
kept hitting was the iteration budget, not the method — see `FINDINGS_GRAD_ARM.md`, which also closes
the gradient-batch hypothesis as a negative once the arms are matched on compute instead of
iterations.

The previous default `cnn` controller (`cnn_v2.pt`) scores **46.91** on the full 5000 and **50.72** on
`ALL[4200:5000]`, a pristine split it never saw for training *or* checkpoint selection (`cnn_PM.pt`
scores 47.87 and 52.31 on the same two). It is a pure function of the observed state, the 5-second
preview, and its own recent actions — **no per-segment memorization**.

## What this benchmark actually measures — read this before the numbers

Measured on 500 pristine segments under `cnn_v2`, the scored window is dominated by near-straight
highway driving:

| regime | % of steps | % of lataccel cost |
|---|---|---|
| `\|τ\| < 0.2` — **essentially straight** | **72.2%** | **61.0%** |
| 0.2 – 0.5 | 12.9% | 14.4% |
| 0.5 – 1.0 | 8.6% | 12.2% |
| 1.0 – 2.0 | 5.7% | 7.9% |
| `\|τ\| > 2.0` — hard corner | 0.7% | 4.6% |

Median `\|target lataccel\|` is **0.075 m/s²** at a median speed of **58 mph**. So 61% of the tracking
cost comes from holding a nearly straight line, and hard cornering — what "lateral control" evokes —
is 0.67% of steps and 4.6% of cost.

**And roughly two thirds of the score is not controllable at all.** The causal floor is **31.24**
(`FINDINGS_FLOOR.md`) against `cnn_v2`'s 46.21 on that split, so ~68% of the achievable number is
irreducible rejection of a synthetic random walk that no causal controller can anticipate. The actual
controller-quality signal is the remaining ~15 points, and most differences between good controllers
here are 1–5 points inside that sliver. That is the honest explanation for why a long sequence of
reasonable ideas in this repo returned nulls.

**What is realistic.** Highway lane-keeping genuinely is mostly straight-line disturbance rejection,
so the task shape is not wrong. Dead time of 200–500 ms is plausible for steering actuation plus
vehicle response, gain rising with speed is real bicycle-model behaviour, and the tracking-vs-jerk
tradeoff and the value of preview are both real.

**What is not.**

- **The disturbance model.** A random walk with white increments (σ ≈ 0.034) is a modelling
  convenience; real lateral disturbances have structure (camber persists, gusts have shape, road
  texture is high-frequency). The causal floor derivation rests on that whiteness, which is a
  property of how the simulator was written rather than of a vehicle.
- **A 16% left/right gain asymmetry** (`FINDINGS_SYSID.md`), measured at ~20σ. Real vehicles are
  near-symmetric; this is almost certainly a learned-model artifact.
- **Gain blowing up above `\|lataccel\| > 2`**, a region holding 0.8% of the data — the plant is
  extrapolating, not reporting physics.
- **Fixed per-segment RNG seeds** (`tinyphysics.py:116`), which make the metric exploitable in a way
  no physical system is.
- **The plant is a learned model of a bicycle model**, two abstraction layers from a vehicle.

**The uncomfortable implication.** Hard corners are where control skill would actually show, and they
are exactly where the simulator is least trustworthy. So the benchmark cannot reward cornering
competence even in principle, and optimising the tail means fitting simulator artifacts.

Read the results below accordingly: **`cnn_v2` at 46.91 is a good benchmark score, demonstrating
competent straight-line noise rejection under a synthetic disturbance model.** It is not evidence
that it would drive a car well. What transfers is the method — differentiable-sim training, matched
controls, the floor derivation, the negative results — rather than the controller.

### The two newest promotions — one accepted, one retracted

**`ff_pi_boot` (52.30 → 51.22) — bootstrapped integrator, mechanism understood.** A feedback
integrator has to *discover* the steady-state steering offset a new target needs by accumulating
error; a trace through a real corner shows error stuck at +0.13…+0.41 for seven steps while steering
creeps 1.129 → 1.146. The plant model already knows the answer, so the integrator is nudged toward
it instead: `integ += boot * (integ_target − integ)`. On `ff_pi_rl2` the feedforward is deliberately
detuned 1.79×, so the right anchor is the *residual* that detuning leaves, `(u_true − ff)/ki`, using
the measured gain rather than the detuned one. Verified −1.080, 95% CI [−1.506, −0.634], 3395/5000
improved. The same mechanism is worth −38% on the stock PID, which has no feedforward at all and so
must discover the entire offset (110.76 → 68.41, 4598/5000, gains untouched).

**`cnn_v2.pt` (47.87 → 46.91) — a better training run, and nothing more interesting than that.**
This slot previously held `cnn_dual.pt` on the strength of a −0.978 improvement over `cnn_PM.pt`,
attributed to its behaviour-cloning→policy-optimisation schedule. **That attribution was wrong**, and
the control run for a later experiment is what caught it. Against a *fresh from-scratch run of the
same architecture and recipe* — the comparison the promotion never made — `cnn_dual` is a coin flip:

| comparison | mean Δ | 95% CI | median | better |
|---|---|---|---|---|
| `cnn_dual` vs `cnn_PM` (the original claim) | −0.978 | [−1.51, −0.54] | −0.202 | 2910/5000 |
| `cnn_dual` vs **fresh matched control** | −0.016 | **[−0.61, +0.39]** | **+0.005** | **2490/5000** |
| fresh control vs `cnn_PM` (pure run variance) | −0.961 | [−1.41, −0.54] | −0.200 | 2889/5000 |

A plain rerun reproduces essentially the *entire* −0.978. So **BC→PO buys nothing over PO from
scratch**; `cnn_PM.pt` was simply a below-average run. The shipped `cnn_v2.pt` is that plain rerun —
statistically tied with `cnn_dual` (46.91 vs 46.89), reproducible from the documented retrain command,
and carrying no unexplained mechanism. `cnn_dual.pt` stays tracked purely as evidence for the negative.

Two things worth taking from this. **Run-to-run training variance on this architecture is ~1.0 point
on the headline and ~1.6 on an 800-segment split** — larger than most effects chased in this repo.
And the original promotion applied both statistical gates *correctly*, bootstrap CI and sign test,
and still reached the wrong conclusion, because it compared a new artifact against an **old artifact**
rather than a **matched control**. Rigour on the wrong comparison is still wrong. Full detail in
`FINDINGS_GAIN_PRIOR.md`.

Checkpoint selection is worth recording as a separate method note, since it is a *different* trap
from the one above and both fired in the same session. Four `dualcnn` checkpoints were compared on
the pristine split, and **three of them improve the mean while making the median segment worse**
(`po_0125`: mean −1.784 but median **+1.093**, only 234/800 better). Those are tail artifacts, not
controllers — the benchmark cost is a mean, so a few chaotic blow-ups moving the right way can
manufacture an "improvement" the typical segment never sees. Only `po_0325` had a negative median
*and* a sign test clear of chance, so it is the one carried forward — and it then turned out to be
tied with a plain rerun anyway. Screening on the median catches the tail artifact; only a matched
control catches run variance. **Both gates are needed, and neither substitutes for the other.**

Two independently-designed 2-DOF controllers (ours and a ported reference) land 0.5 apart at ~59 —
indistinguishable at this metric's noise level — which is what pins the **classical
feedforward+feedback plateau** on this cost landscape. The learned net clears it by ~11 points,
a margin well outside the noise.

**Where that margin comes from — two different answers, before and after the tail is fixed.**

Against `ff_pi_tuned`, the margin is almost entirely a tail. On the clean 500-segment split
`ff_pi_tuned` scores 55.68 and `cnn` 48.14, and tracing every segment individually shows that
**six segments out of 500 account for 72% of that 7.54-point gap** — `cnn` averages 142 on those six
against the classical controller's 596. They are low-speed, large-lateral-acceleration corners where
the classical controller's loss arrives in short bursts (71% of its squared error in 10% of the
timesteps).

Against `ff_pi_rl2`, which removes that tail, **the remaining gap is broad**. Over 3000 segments:

| | `ff_pi_tuned` | `ff_pi_rl2` | `cnn` |
|---|---|---|---|
| median | 46.45 | 46.45 | **44.33** |
| p90 | 79.72 | 79.67 | **75.91** |
| p99 | 297.91 | 237.69 | **150.13** |
| max | 1994.78 | 906.16 | **761.98** |
| worst 10% share of cost | 29.3% | 26.8% | **23.6%** |

`cnn` wins at **every** quantile, median included, and six segments now hold only 19.5% of the gap
while the median difference alone accounts for 29%. So "not broad superiority" was true of the
comparison against `ff_pi_tuned` and is **not** true in general — an earlier revision of this section
overstated the concentration, and the anti-windup fix is what made the difference visible by removing
the tail that had been masking it.

Note also that `ff_pi_rl2`'s median and p90 are *identical* to `ff_pi_tuned`'s: the fix is purely
tail-targeted, with the entire gain in p99 and max, exactly as intended.

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

**A causal controller cannot average below ~31 on this cost**, which is what makes the top of the
leaderboard legible. Derived from the plant's own measured noise (`FINDINGS_FLOOR.md`): TinyPhysics
samples each step from a predictive distribution with `E[sigma^2] = 0.001174`, and those shocks are
white (increment autocorrelation +0.042), so they can only be reacted to, never anticipated. A shock
is corrected over the impulse response `H`, leaving `sum (1-cumH)^2 = 3.3228` uncorrected, giving a
lataccel floor of **19.50**; holding the wheel still costs **11.74** in jerk. Since
`min(A+B) >= min(A) + min(B)`, the causal total is bounded below by **31.24** — and it is a strict,
unreachable bound, since those two minima sit at opposite operating points.

Three independent routes agree: this derivation (31.24), trajectory optimization through the
differentiable plant (29–34), and the best honest leaderboard entries (~36).

That places the bands precisely. **~31 is an unreachable bound, ~36 is the honest frontier, and the
7–30 band is not controllers at all** — it is offline action sequences optimized against the public
set's *fixed per-segment RNG seeds*, where the "noise" is a known sequence rather than noise. We
demonstrated the mechanism from the other side: replaying even a good controller's own actions
open-loop scores **~970** against **~54** closed-loop, because only feedback rejects drift.

The floor also says where the remaining headroom is. Our best sits ~1.2x off the tradeoff-optimal
jerk but **1.37x** off the lataccel floor — roughly twice as much proportional room in *tracking* as
in *smoothness*, so further smoothing is the wrong direction.

Our 47.87 is an honest, generalizing controller that sits **mid-pack among the honest entries**: it
clears the classical plateau by ~11 points and is at or slightly ahead of the well-known ML_PID
entry (50.63) — though per caveat 2 above, a few points is within subset uncertainty, so treat that
particular comparison as a tie rather than a win. A handful of MPC- and PPO-based controllers score
clearly lower. **The honest frontier is ~36**, not ~50 — reaching it is a method change (see below),
not a tuning gap.

Why the exploits need the fixed seed, measured directly. `tinyphysics.py:116` seeds the RNG from
`md5(filename)`, so **every segment has exactly one noise realization, on every run, forever**. The
plant is not stochastic per segment — it is a deterministic function of the action sequence. Record
a good controller's actions and replay them **open-loop, with no feedback at all**:

| | mean cost, 8 segments |
|---|---|
| closed-loop (with feedback) | 43.03 |
| open-loop replay, **same** (fixed) seed | **43.03** — max difference `0.00e+00` |
| open-loop replay, **different** noise draw | 238.47 (6× worse) |

Bit-exact on the seed it was recorded against; catastrophic on any other. That gap is the entire
exploit. An earlier revision of this section reported the ~970-vs-~54 figure without the
qualifier, which read as "open-loop replay fails" — it does not; it fails only against a noise
realization it was not tuned for.

So the exploit converts control into **offline trajectory optimization**: optimize an action
sequence against a segment's known shock sequence, fingerprint the segment at eval time (the first
100 steps are given), and replay. A *known* future disturbance can be pre-compensated — cancel the
shock at t+5 with an action at t — which removes the dead-time penalty that produces the 31.24
causal floor in the first place. The same principle in a different wrapper is "online sim probing
with RNG reset": call the simulator inside `update()`, save/restore `np.random.get_state()`, and
pick the best action after seeing the actual future.

These are lookup tables keyed to a random seed, not functions of observations.

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
| **Measured gain prior for the `cnn`** (cfg `G`: preview window divided by `G(v)`, so the conv sees steering units) | Null (+0.25 headline, CI [−0.10, +0.60]; +0.15 pristine) | The bootstrap's payoff decays monotonically with how much physics the host already has: −38% on stock PID (no feedforward), −1.08 on `ff_pi_rl2` (explicit ff + gain schedule), **null** on the `cnn`, whose `film` layer has already learned the schedule. It led on both the torch surrogate *and* the selection split and lost both untouched splits — winner's curse. |
| **Behaviour cloning → policy optimisation** (`dual_train.py`) | Null vs a matched control (−0.016, CI [−0.61, +0.39]) | Reproduced by a plain rerun; the apparent gain was run-to-run training variance. |
| **Denoising the feedback** | Not applicable | This is **process** noise on a **fully-observed** state — the sampled lataccel *is* the car's real position and is what gets scored. There is no clean signal hiding underneath to recover. |

The remaining gap to the ~36 frontier is a **method** difference — MPC on a smooth linear plant
model, or value-based RL (PPO) — not another lever on this approach.

### Negatives from the lag/anticipation line

All measured against `pid_smooth` (80.72) or `ff_pi_rl2` (52.30) on held-out splits.

| Tried | Outcome | Why |
|---|---|---|
| **Smith predictor** (`pid_lag.py`) | −16 | Substitutes a model prediction into the feedback, which is known to degrade *disturbance* rejection — fatal on a plant whose disturbance is a random walk with lag-1 autocorrelation 0.98. |
| **Latch the reference until the plant reaches it** (`pid_hold.py`, `mode='latch'`) | Catastrophic | Makes the reference a staircase; step changes inject exactly the high-frequency content the jerk term charges at 10000×. |
| **Reference governor** (`pid_hold.py`, `mode='rate'`) | +1.658, CI [−0.740, +4.189] | Knife-edge overfit: only 39/1000 segments improved on clean data. |
| **Pending-response correction** (`pid_pend.py`) | `pred_i` +2.2…+16.7, `pred_p` +0.14…+2.1 | Discounting error that in-flight commands will fix removes *integral* action, which is the load-bearing element here. Predicted `pred_i` would help and `pred_p` would hurt — exactly inverted. |
| **Lookahead on `ff_pi`'s feedback** (`ff_pi_look.py`) and **velocity-scheduled ff lead** (`ff_pi_vlead.py`) | Both null | Anticipation is a single resource: `ff_pi` already anticipates, so a second mechanism has nothing left to buy. |
| **Speed-scheduled averaging weights on `pid_w_ff`** (`pid_wff_v.py`) | Null in both directions | The upstream `[5,6,7,8]` weights have center of mass 44/26 = 1.692 steps, which *is* the optimum of the unscheduled sweep. Scheduling that center by speed loses whichever way it moves. |
| **Bootstrapping the P and D paths** | Null (best variant −0.07, non-monotone) | The bootstrap fixes a *stateful* element — an accumulated value that must be discovered and can therefore be wrong. `kp·e` and `kd·Δe` are recomputed from scratch each step, so there is no referent to anchor. The nearest analogues are feedforward (tested, harmful) and rate feedforward (which the lookahead already is). |
| **Conditional integration for the `cnn`** | No headroom — not built | The mechanism worth ~2 points on `ff_pi_rl2` needs the plant's rate clamp to bind. Under `cnn` it binds **1 step in 24,000** (0.004%) and the integrator sits at its ±5 clip 0.02% of the time. |

Two general principles came out of this line and both held up under repeated test:
**anticipation is a single resource** (four independent nulls once a feedforward exists), and
**integral action is load-bearing on this plant** — help it converge, never discount it.

## Unknowns

Stated explicitly rather than papered over, because in each case the effect is measured but the
explanation is not.

- ~~**Why behaviour cloning → policy optimisation beats policy optimisation from scratch.**~~
  **Resolved: it doesn't.** Against a fresh matched control the BC→PO advantage is a coin flip
  (−0.016, CI [−0.61, +0.39], 2490/5000). The effect was training run-to-run variance throughout.
  Two signals had already pointed this way and were noted without being acted on — a geometry probe
  refuted the "BC flat-minimum" account, and iterating BC→PO did not compound, which a genuine
  basin-escape mechanism would have done. See the results table above and `FINDINGS_GAIN_PRIOR.md`.
- **Whether `gain_scale` and `ki` should be retuned with the bootstrap active.** `ff_pi_rl2`'s gains
  were co-tuned on the assumption the integrator discovers the residual by accumulation; the
  bootstrap changes that assumption. Retuning was not attempted because gain tuning on this family
  has overfit every previous attempt (86.96 vs 81.40 on the PID stack), and it would need
  `sigma=0.15` plus the held-out guard to be trustworthy.
- ~~**Why the derivative term helps at all.**~~ **Resolved: a negative `d` is not a derivative, it is
  a low-pass filter.** In comma's discrete form `u = p·e_t + d·(e_t − e_{t−1}) = (p+d)·e_t − d·e_{t−1}`,
  so `p=0.195, d=−0.053` is the two-tap FIR `0.142·e_t + 0.053·e_{t−1}` — both taps positive, summing
  to `p`. DC gain is identical for every `d`, so steady-state tracking is untouched; only the
  high-frequency path changes. At 5 Hz (Nyquist) `d=−0.053` gives |H| = 0.089 against a DC gain of
  0.195 (a ratio of 0.46, i.e. 54% attenuation), `d=0` is flat, and Ziegler-Nichols' `d=+0.60`
  *amplifies* 7.15× — which is the visible jitter in `step_response.png` and part of why ZN scores
  3198. The right sign is negative because the error carries the plant's noise, whose increments are
  white and therefore unpredictable, while the plant has 2–4 steps of dead time: a correction issued
  against high-frequency disturbance arrives after that disturbance has changed. Consistent with the
  measurement, the gain shows up in **lataccel** (44.67 vs 45.04 at `d=0`) more than in jerk (21.01
  vs 21.09) — the mechanism is "stop issuing late corrections against unpredictable noise", not
  "less jerk". Same principle as the Tikhonov reference, `ff_pi`'s 1.79× detuning and the ~2-step
  lookahead optimum: on a dead-time plant with white disturbance, react less.
- **Absolute scores carry several points of subset uncertainty** (see caveat 2 above). The ratio to
  PID is the robust statistic; small cross-entry gaps on the leaderboard are not resolvable.

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
| `cnn_v3.pt` | **Default** `cnn` weights (45.74) — same recipe as `cnn_v2`, trained to convergence |
| `cnn_v2.pt` | Previous default (46.91) — plain policy optimisation, stopped at 400 iterations |
| `cnn_dual.pt` | BC→PO weights (46.89); kept as evidence the schedule adds nothing over a matched control |
| `cnn_PM.pt` | v1 weights (47.87), tag `v1-learned-47.87`; a below-average run, kept for reproducibility |
| `dual_train.py`, `FINDINGS_GAIN_PRIOR.md` | Two-phase BC→PO trainer, and the writeup showing it is a null |
| `controllers/ff_pi.py` | 2-DOF feedforward + PI baseline |
| `controllers/ff_pi_tuned.py`, `controllers/pid_tuned.py` | CMA-ES-tuned variants; parameterised copies so the quoted baselines stay untouched |
| `controllers/ff_pi_rl2.py` | Conditional anti-windup against the plant's lataccel rate clamp (52.30) |
| `controllers/ff_pi_boot.py` | **Best classical** (51.22) — `ff_pi_rl2` + integrator bootstrapped to the model residual |
| `controllers/pid_phys.py`, `pid_smooth.py`, `pid_boot.py` | The PID stack: velocity-scheduled lookahead from the measured step response, Tikhonov smoothing, bootstrapped integrator (110.76 → 68.41) |
| `controllers/pid_lag.py`, `pid_look.py`, `pid_hold.py`, `pid_pend.py`, `pid_wff_v.py`, `ff_pi_look.py`, `ff_pi_vlead.py` | Documented negatives from the lag/anticipation line; each reproduces its parent exactly at default parameters |
| `FINDINGS_FLOOR.md` | The causal cost floor (31.24) derived from the plant's noise, and why sub-30 entries are seed exploits |
| `FINDINGS_PID_LAG.md`, `FINDINGS_CLAMP.md`, `FINDINGS_GAIN_PRIOR.md`, `CYNIC_REVIEW.md` | Full measurement logs and the adversarial review |
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
