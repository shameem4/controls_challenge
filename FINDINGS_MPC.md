# The MPC pivot: what happened, and why the learned policy still wins

*Recorded on branch `mpc-lpv`, last updated 2026-07-25. Headline numbers quoted here are those current at the time; `master` has since moved on — see the README for the figures that stand today.*

Branch `mpc-lpv`. Goal was sub-40 (the honest leaderboard frontier is ~36, an MPC on a linear
LPV-ARX model). **Not achieved.** `master` remains the deliverable at 47.87 (`v1-learned-47.87`).

This is written up because the negative results are more useful than the attempt.

## Verdict

Properly-sized evaluation on clean segments (never used for identification, DAgger, or tuning):

| set | `cnn` (policy) | GN-MPC (best rdu) |
|---|---|---|
| `[500:620]`, 120 segs | **53.58** | 67.12 |
| `[500:540]`, 40 segs | **61.04** | 78.08 |

An earlier 10-segment result (MPC 45.56 vs cnn 48.15) did **not** survive; it was small-sample
noise, and the `rdu` optimum moved between ranges (10k → 30k), confirming that tuning was overfit.

## The bug that invalidated every earlier MPC attempt

The plant clamps its output slew to `MAX_ACC_DELTA`. Representing that as a hard clamp *inside the
planning model* is fatal:

```python
pred = prev + (pred - prev).clamp(-MAX_ACC_DELTA, MAX_ACC_DELTA)   # d pred/d u == 0 when saturated
```

Once saturated the Jacobian columns vanish, so the optimiser is blind: actions run to the ±2 rails,
the plant sits in a permanent ±0.5 square wave, and it cannot escape. Caught in a closed-loop trace
(lataccel alternating 0.427 / 0.927 — exactly `MAX_ACC_DELTA` apart — with planned == realised to
`0.00e+00`, which ruled out an execution bug).

Removing it from the *planning* model (the plant still applies it), with the MPC controlling the
surrogate — a plant it models exactly:

| planning clamp | cost |
|---|---|
| `hard` | 2471.96 |
| `soft` (tanh) | 1572.37 |
| **`none`** | **6.17** |

6.17 is about the analytic optimum (the injection exploits score 6.88). **The formulation, the
Gauss-Newton solver and the cost were correct all along.** The earlier LPV-MPC failure was the dual
form of the same bug: it ignored the clamp entirely and planned trajectories the plant then clamped.

## Model quality was improved a lot, and it wasn't enough

Multi-step free-run error vs the *plant-twin floor* (the true plant re-run from the same state, so
its only error is fresh sampling noise — i.e. the information-theoretic limit):

| H | surrogate | LPV | plant-twin floor | ratio (sur) | ratio (LPV) |
|---|---|---|---|---|---|
| 1 | 0.0418 | 0.0615 | 0.0426 | 0.98 | 1.44 |
| 10 | 0.1651 | 0.3298 | 0.1425 | 1.16 | 2.31 |
| 30 | 0.2771 | 0.4585 | 0.2085 | **1.33** | 2.20 |

The surrogate distils the plant's *exact conditional mean* (queryable via `mode='expected'`, so
targets are noise-free). At H=1 it beats the sampling twin, which is correct — the conditional mean
is the RMS-optimal predictor.

Attempts that did **not** help: higher-order ARX, and fitting ARX directly on 25-step free-run error
(ratio 2.05 → 2.05). Two model classes × two fitting objectives hit the same wall, so ~2.2x is the
practical limit of *linear* modelling here; the residual is nonlinearity an ARX cannot represent.

DAgger **did** help: retraining on the states the MPC actually visits took surrogate val RMSE
0.181 → 0.153 and the MPC 50.36 → 45.56 on its tuning range.

## Why the policy wins — the actual conclusion

**This is a model-fidelity story, not a noise-structure story.** An earlier draft of this document
blamed certainty equivalence — MPC plans against the mean, so (the argument went) it must be
miscalibrated on a noisy plant. **That explanation was tested and refuted.**

It only holds for noise that multiplies the *control* signal, the case where the LQG separation
principle provably fails and the optimal gain is strictly attenuated (Kleinman 1969, *IEEE TAC*
14(6); Todorov 2005, *Neural Computation* 17(5); independently Brainard 1967 in economics).
Measuring TinyPhysics directly:

- holding the entire history fixed and forcing the current action from 0 → 2.0, the plant's
  output-distribution std stays flat at **~0.030** (non-monotonic, ±10%);
- on-policy, `corr(|steer|, output std) = -0.20` — slightly *negative*; std falls from 0.030 to
  0.025 as |steer| grows.

The noise is **control-independent (additive)**, so certainty equivalence should hold and MPC-on-
the-mean is not structurally handicapped here. (This is also a way the simulator differs from real
driving: Nash & Cole 2019 identified genuinely signal-dependent noise in *human* steering,
RMS(δ)/W ≈ 0.57.)

The correct explanation is simpler and better supported: **MPC is only as good as its model, and the
policy has no model to be wrong.** MPC plans over a horizon *through* the model, so model error
compounds. Three measurements pick this over any noise-structure account:

1. MPC on a plant it models *exactly* (the surrogate) scores **6.17**, about the analytic optimum —
   the planner is sound and certainty equivalence is not the problem;
2. swapping in the real plant and changing nothing else collapses it to ~78 — the only difference is
   model fidelity;
3. improving the model moves the score monotonically: one DAgger round took the surrogate val
   0.181 → 0.153 and the MPC 50.36 → 45.56.

The `cnn` policy maps state → action directly and uses **no model at run time**, so model error
cannot touch it at all.

**Transferable lesson: here a model-free policy trained on the true plant beats model-based
planning, because planning inherits the model's error and the policy does not.** For MPC to win it
would need fidelity close to the information floor — and part of our 1.33x is irreducible, since
even a perfect model has 0.21 lataccel error at H=30.

## If someone wants to continue

- more DAgger rounds and much more surrogate data (it still overfits: train 0.015 vs val 0.153)
- more model fidelity is the whole game: the measured chain is 6.17 (exact model) -> ~78 (real
  model), and DAgger moved it monotonically. Note the noise is additive, so risk-aware / tube MPC
  is NOT indicated here — that fix targets control-multiplicative noise, which this plant lacks
- the MPC is slow (~90 Jacobian backprops per control step); a full 5000-segment eval is hours, so
  any real submission would need it distilled into a policy

## Files

`lpv_id.py` `lpv_id2.py` (identification) · `lpv_validate.py` `lpv_floor.py` `surrogate_gate.py`
(the prediction gates) · `surrogate.py` `dagger.py` (surrogate + DAgger) · `controllers/mpc_lpv.py`
(linear MPC) · `mpc_sur.py` (gradient MPC) · `mpc_gn.py` (Gauss-Newton MPC, the working one)
