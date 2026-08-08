# How much future does the controller get, and would more help? 5 s, and no.

## What the harness provides

| | steps | seconds |
|---|---|---|
| `FUTURE_PLAN_STEPS` (`tinyphysics.py:38`, `FPS * 5`) | 50 | **5.0** |
| what `ff_pi`/`ff_pi_tau` consume (whole preview, inside the Tikhonov solve) | 50 | 5.0 |
| what the CNN consumes (`nets.H`) | 25 | 2.5 |
| plant context length (past) | 20 | 2.0 |

## How far ahead can preview possibly matter?

The cost-optimal reference is the Tikhonov solve `c* = (I + lam D'D)^-1 tau`. That inverse is a
Green's function decaying as `r^k`, with `r` the root of `lam r^2 - (1 + 2 lam) r + lam = 0`. At
`lam = W_jerk / W_track = 2.0048`, **r = 0.5004** -- the influence of the target `k` steps ahead
halves every single step.

| k steps | seconds | analytic influence | measured change in c*[now] vs full preview |
|---|---|---|---|
| 1 | 0.1 | 5.0e-1 | 0.96% of target scale |
| 3 | 0.3 | 1.3e-1 | 0.24% |
| 5 | 0.5 | 3.1e-2 | 0.061% |
| 10 | 1.0 | 9.8e-4 | **0.0019%** |
| 20 | 2.0 | 9.7e-7 | 1.7e-8 (numerically zero) |
| 50 | 5.0 | 9.2e-16 | — |

Measured by recomputing the optimal reference with the preview truncated at `k` and comparing to the
full-preview answer, over 200 pristine segments and ~7000 sample points.

**Preview beyond about 1 second is mathematically worthless**, and beyond 2 seconds it is zero to
numerical precision. The harness already supplies 5x more than the cost function can use.

Adding the plant's own dynamics does not change this: dead time is ~0.25 s and the step response
settles in ~10 steps, so the total relevant horizon is roughly the smoothing kernel (10 steps) plus
settling (10 steps) = ~2 s. Still well inside what is provided.

## Why the horizon is short: it is set by lam

`r` rises steeply with the jerk weight:

| lam | r | half-life |
|---|---|---|
| 2 (this benchmark) | 0.500 | 1 step |
| 10 | 0.730 | 2.2 steps |
| 100 | 0.905 | 6.9 steps |

A jerk-dominated cost would genuinely need long preview. This one does not -- the tracking term
dominates enough that the optimal trajectory is nearly local.

## Consistent with the one experiment already on record

`0aeaaad` tested the CNN at preview 25 -> 45 steps in a from-scratch A/B with identical data,
curriculum, schedule and iterations: **real sim 50.754 vs 49.750, +1.004, CI [-0.71, +2.57]** -- a
null. At the time that was recorded as one of the cnn negatives without explanation. The Green's
function above is the explanation: the extra 20 steps carry an influence of order 1e-6.

## Conclusion

Modifying the sim to provide longer preview would change nothing. The binding constraint is not how
far ahead the controller can see -- it already sees 5x further than matters -- but the two limits
already established: the Bode waterbed under dead time (`FINDINGS_SPECTRUM.md`) and the
unpredictability of the plant's innovation (`FINDINGS_NOISE_PREDICT.md`).
