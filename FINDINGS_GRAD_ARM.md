# The gradient arm, run to convergence: hypothesis dead, checkpoints were undertrained

## What was open

`0aeaaad` left one non-null cnn result — raising the gradient accumulation `ACC` from 4 to 12
(effective batch 32 → 96) scored −1.269 [−2.26, −0.61]. Its own commit flagged three defects:

1. arms matched on **iterations, not compute**, so `ACC=12` also got 3× the data per iteration;
2. it **never converged** — the best checkpoint was the final iteration, still improving;
3. it **lost to released `cnn_PM` on `ALL[:5000]`** (48.27 vs 47.87).

This rerun fixes (1) and (2): both arms perform **10,800 rollouts** (grad 900 iters × ACC 12; base
2700 iters × ACC 4), from scratch, identical pool/curriculum/schedule, checkpoints selected on the
real numpy sim over `ALL[4200:4700]`.

## The A/B: the gradient batch does not help

| split | grad12 (ACC=12) − base4 (ACC=4) | median Δ | better |
|---|---|---|---|
| `ALL[5000:6000]` n=1000 | **+1.461 [+0.59, +2.77]** | +0.119 | 458/1000 (z=−2.7) |
| `ALL[6000:8000]` n=2000 | −0.123 [−2.02, +0.97] | +0.148 | 892/2000 (z=−4.8) |
| `ALL[:5000]` n=5000 | **+0.765 [+0.52, +1.04]** | +0.079 | 2341/4999 (z=−4.5) |

At matched compute the larger gradient batch is **neutral to harmful** — positive mean deltas with
CIs excluding zero on two splits, and sign tests against it on all three. The original −1.269 was the
confound its own commit predicted: 3× data per iteration, credited to gradient-variance reduction.
**Twelfth negative confirmed as a negative.** The hypothesis is closed.

## The control arm is a new best: the released checkpoints were undertrained

`base4` is nothing but the standard recipe (`ACC=4`) run 6.75× longer than the released checkpoints —
2700 iterations / 10,800 rollouts against 400 iterations / 1,600 rollouts.

| | `ALL[:5000]` (headline) | `ALL[5000:6000]` | `ALL[6000:8000]` |
|---|---|---|---|
| **`cnn_v3` (base4)** | **45.744** / med 43.98 / p99 132.3 | **45.648** / 43.60 / **123.9** | **48.208** / 43.98 / 127.9 |
| `cnn_v2` | 46.911 / 44.14 / 148.2 | 47.361 / 43.94 / 133.8 | 49.273 / 44.64 / 138.9 |
| `cnn_dual` | 46.894 / 44.32 / 151.6 | 47.843 / 44.30 / 147.3 | 49.451 / 44.52 / 138.8 |
| `cnn_PM` | 47.872 / 44.49 / 150.2 | 49.795 / 44.06 / 151.2 | 50.134 / 44.98 / 139.3 |

Versus `cnn_v2`:

| split | mean Δ [95% CI] | median Δ | better |
|---|---|---|---|
| `ALL[:5000]` | **−1.167 [−1.91, −0.64]** | −0.364 | 3016/5000 (z=+14.6) |
| `ALL[5000:6000]` | **−1.713 [−2.55, −1.03]** | −0.361 | 619/1000 (z=+7.5) |
| `ALL[6000:8000]` | −1.065 [−3.26, +1.25] | −0.428 | 1251/2000 (z=+11.2) |

**This is the first cnn intervention in the project that improves mean, median, p99 and per-segment
win-rate simultaneously.** Every previous one — mining, headroom selection, ensembles — bought tail
at the cost of median. The effect is well above the ~1.0-point run-to-run training variance, and it
replicates on three splits including the 5000-segment headline.

The cause is mundane and slightly embarrassing: **the models were stopped too early.** The "~48
ceiling" that eleven interventions kept hitting was not a property of the method. It was the
iteration budget. `15fa575` even documented the mechanism that hid this — "at iteration 50 the wide
model was ~2× better … that was learning SPEED, not final quality" — and the same reasoning applies
in reverse to a fixed 400-iteration budget: every arm was being compared at a point none of them had
converged to.

## Promoted

`cnn_v3.pt` (= `ckpts/cap_base4_02625.pt`, md5 `775a6fd6…`) is the new default in
`controllers/cnn.py`. Headline **45.744** on `ALL[:5000]`, from 46.911.

## What this does and does not reopen

It does **not** reopen the eleven negatives. Capacity, preview, mining, BC, distillation, ensembles,
routing and the rest were all A/B'd against *matched* controls at the same budget, so their
conclusions stand relative to their controls — a longer budget lifts both arms.

It does mean the **absolute** ceiling claim was wrong, and the natural follow-up is simply: does it
keep going? `base4`'s real-sim selection picked iteration 2625 of 2700 — again near the end, again
possibly stopped mid-descent. The honest next experiment is a longer run still, with a stopping rule
based on validation plateau rather than a fixed count.

## Operational note

This box rebooted four times during these runs, twice killing a training silently. The `cap_ab.py`
resume path (state written every `SAVE_EVERY` iterations, weights + optimiser + iteration + RNG)
saved both arms; the gradient arm resumed at iteration 476 and the baseline at 2476. Resume needed
`torch.load(..., weights_only=False)` because the numpy RNG state will not unpickle under the newer
default.
