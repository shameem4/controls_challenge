# The causal cost floor, derived from the plant's own noise

What is the best score any controller can achieve *without knowing the future noise realization*?
This gives a target to measure against, and it sets the score below which an entry cannot be an
honest controller.

## The measurement the whole thing rests on

TinyPhysics is autoregressive: each step it emits a distribution over the next lataccel token and
samples from it (temperature 0.8). The deviation of the sample from that distribution's mean is the
**unpredictable shock** — no controller can anticipate it, because it has not been drawn yet.

Measured directly from the plant's predictive distribution rather than inferred from residuals
(60 segments, 24 000 scored steps, under `ff_pi_boot`):

```
  E[sigma^2]   = 0.001174          <- the quantity the floor needs
  RMS sigma    = 0.03426
  median sigma = 0.03359           (p10 0.01842, p90 0.04460 -- it is state-dependent)
  token bin    = 0.00978           quantization floor, well below sigma
```

This independently confirms the 0.0337 used in earlier work (median agrees to 0.3%). Because the
floor scales with `sigma^2`, the correctly-weighted `E[sigma^2]` is what must be used, not the mean
or median of sigma — using the mean would understate the floor by ~9%.

The shocks are **white** (increment autocorrelation +0.042); it is the *level* that has
autocorrelation 0.98. White shocks cannot be predicted, only reacted to.

## Lataccel floor: 19.50

A shock lands. The state is fully observed, so the controller can begin correcting on the very next
step — but the correction is *delivered* over the plant's impulse response
`H = [0.00, 0.02, 0.08, 0.22, 0.38, 0.30]`, so a fraction stays uncorrected meanwhile:

```
  k    cum H     residual (1-cum)   residual^2
  0    0.000         1.000            1.0000
  1    0.020         0.980            0.9604
  2    0.100         0.900            0.8100
  3    0.320         0.680            0.4624
  4    0.700         0.300            0.0900
  5    1.000         0.000            0.0000
                                sum = 3.3228
```

    lataccel floor = 5000 * E[sigma^2] * 3.3228 = 19.50      (as a contribution to total_cost)

## Jerk floor: 11.74

Minimizing jerk alone means holding the wheel still and paying only the shocks:

    jerk floor = 10000 * E[sigma^2] = 11.74

Note this is *not* the 16.52 figure that applies at the cost-optimal tradeoff (tracking the Tikhonov
reference `c*`, whose own jerk is 5.16). 11.74 is the unconstrained minimum of the jerk term, which
is what a lower bound on the sum requires.

## Causal total lower bound: 31.24

Since `min(A+B) >= min(A) + min(B)`:

    total_floor >= 19.50 + 11.74 = 31.24

The two minima are attained at *different* operating points — perfect tracking has jerk 11.31, and
holding still has lataccel ~1030 — so no controller reaches 31.24. It is a strict lower bound, and a
loose one.

### Cross-checks

Three independent routes agree, which is the main reason to trust the number:

| route | value |
|---|---|
| this derivation (plant noise + impulse response) | 31.24 |
| trajectory optimization through the differentiable plant | 29–34 |
| best honest leaderboard entries | ~36 |

## Where we actually stand

```
                    total   lataccel   (x floor)   jerk
  causal bound      31.24     19.50      1.00x    11.74
  honest frontier   ~36          -           -        -
  cnn_v2 (ours)     46.91     26.81      1.37x    20.10
  ff_pi_boot        51.22     31.30      1.61x    19.92
  stock PID        110.76     85.25      4.37x    25.51
```

**The asymmetry is the actionable result.** Against the tradeoff-optimal references, the best
controller sits ~1.2x off on jerk but ~1.37x off on lataccel. There is roughly twice as much
proportional headroom in **tracking** as in **smoothness**, so further smoothing work is the wrong
direction — consistent with `pid_wff_v`, whose only real gain was extra smoothing and was worth ~1.6
on a controller 7 points off the pace.

## What this says about sub-30 scores

A causal controller cannot average below ~31 on this cost. Entries scoring 7–30 are therefore not
controllers in the ordinary sense; they exploit the fact that the public dataset's per-segment RNG
seeds are **fixed**, so the "noise" is not noise to them — it is a known sequence that can be
cancelled by an action sequence optimized offline against that exact realization. The leaderboard's
own entry descriptions say as much ("online sim probing with RNG reset", per-segment action replay).

This repo already demonstrated the mechanism from the other side: replaying even a *good*
controller's own actions open-loop scores **~970** versus **~54** closed-loop, because only feedback
can counteract drift. Per-segment optimized actions only work against the noise realization they
were tuned for.

So the honest range is **~31 (unreachable bound) → ~36 (best honest entries) → 46.9 (ours)**, and
the sub-30 band is a different game.

### Caveats, because this bound is load-bearing

- **It scales with `sigma^2`.** A 10% error in sigma moves the bound ~19%. Sigma is measured directly
  from the plant's output distribution, which is the strongest form available here, but it is
  measured under one controller's state distribution (`ff_pi_boot`) and sigma is state-dependent
  (p10 0.018, p90 0.045). A controller visiting harder states would see a larger sigma and a higher
  floor, so this is, if anything, conservative for good controllers.
- **`H` is a measured impulse response** with its own uncertainty; the `G(v)` fit it accompanies is
  accurate to ±9%.
- **The lataccel floor assumes an idealized corrector** that identifies each shock perfectly and
  issues an exactly-cancelling action. Real controllers cannot instantly separate "a shock landed"
  from "the target moved", and correcting through an imperfect `G(v)` injects error. So 19.50 is
  optimistic and the achievable causal number is above it — our 1.37x is closer than it looks.
- **It ignores the rate clamp**, which binds rarely (1 step in 24 000 under `cnn`) but non-zero.
- The bound is on the **average over segments**; individual segments vary widely.
