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

---

# The same question backwards: how much HISTORY matters?

Even tighter than preview, because here the cutoff is exact rather than asymptotic.

## The plant is exactly 20-step Markov

`tinyphysics.py:132` passes `state_history[-CONTEXT_LENGTH:]`, `action_history[-CONTEXT_LENGTH:]` and
`current_lataccel_history[-CONTEXT_LENGTH:]`. Anything older is structurally invisible to the plant.

Verified rather than assumed -- perturb one past action by 0.5 and measure the change in the next
conditional mean:

| perturb at lag | change in next mu |
|---|---|
| 1 | 1.886e-01 |
| 5 | 3.839e-01 |
| 10 | 2.459e-01 |
| 19 | 9.519e-02 |
| 20 | 7.813e-02 |
| **21** | **0.000e+00** |
| 25 | 0.000e+00 |
| 35 | 0.000e+00 |

So no controller can gain anything from observations older than 2.0 s: they cannot influence what the
plant does next, by construction.

## And the controller needs far less than 20

The Tikhonov Green's function is symmetric, so past targets decay at the same `r = 0.5004`. Sweeping
`ff_pi_tau`'s past-target smoothing window on the tuning split (n=400):

| past | 0 | 1 | 2 | 3 | 5 | 10 | 20 (default) | 40 |
|---|---|---|---|---|---|---|---|---|
| mean | 49.851 | 50.891 | **49.614** | 49.643 | 49.859 | 49.800 | 49.871 | 49.851 |
| median | 46.31 | 46.40 | 45.26 | 45.08 | 45.89 | 46.27 | 46.09 | 46.31 |

Flat from 2 to 40 -- a 0.3 spread across a 20x range of window length, which is noise. Two or three
past targets carry the entire benefit.

## The one exception: the integrator

The integrator has unbounded memory by design, and this project's lag/anticipation line closed with
"integral action is load-bearing on this plant -- help it converge, never discount it" (four
independent nulls from removing or discounting it).

That is not a contradiction. The integrator is not using old observations to predict the plant; it is
estimating a *persistent* quantity -- the residual the detuned feedforward leaves behind, which is why
`ff_pi_boot`'s bootstrap (seeding the integrator with `(u_true - ff)/ki`) is worth what it is. Slow
bias estimation needs long memory; dynamics prediction does not.

## Unified statement

The problem is local in time in both directions:

| | horizon that matters | what sets it |
|---|---|---|
| future | ~10 steps (1.0 s) | Tikhonov decay r = 0.50 at lam = 2 |
| past, for the plant | exactly 20 steps (2.0 s) | `CONTEXT_LENGTH`, a hard cutoff |
| past, for the reference | ~3 steps (0.3 s) | the same Tikhonov decay |
| past, for bias estimation | unbounded | the integrator, estimating a persistent offset |

The harness gives 5 s of preview and 2 s of plant history. Both are already more than the cost
function and the dynamics can use. Neither is a lever.
