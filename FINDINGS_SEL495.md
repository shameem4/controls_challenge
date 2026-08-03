# Retraining `cnn_dual` on the 495 segments where `cnn_v2` wins

## Setup

`cnn_dual` beats `cnn_v2` on 505 of 1000 pristine segments and loses on 495. The proposal: fine-tune
`cnn_dual` on those 495 to fix its weak set.

Design, with the control built in from the start:

| arm | training set |
|---|---|
| **selected** | the 495 segments of `ALL[5000:6000]` where `cnn_v2` wins or ties |
| **matched control** | 495 segments drawn at random from the *same* pool `ALL[5000:6000]` (seed 7, 250 overlap) |

Both warm-started from `cnn_dual.pt`, both `SEED=0`, both 300 iterations of soft-token Gumbel policy
optimisation at full horizon, checkpointed every 25. The only difference is *which* 495 segments.
The control is the point: without it, any change is unattributable to the selection criterion.

Because the training data comes from `ALL[5000:6000]`, that split is burned. Checkpoints were
selected on `ALL[4200:4700]` (disjoint from the `dual_train` pool `2000:4000` and its val
`4000:4200`), and the final comparison is on `ALL[6000:8000]`, never trained on or selected on.

## Checkpoint selection, real sim, `ALL[4200:4700]` n=500

| | best checkpoint | mean |
|---|---|---|
| `cnn_dual` baseline | — | **45.960** |
| selected arm | it125 | 45.917 |
| matched control | it250 | 46.270 |

**No checkpoint of either arm beat the baseline meaningfully** — the best of 13 selected checkpoints
ties it (45.917 vs 45.960), which after selecting the minimum of 13 noisy values is worse than a tie.

## Holdout `ALL[6000:8000]`, n=2000

| | mean | median | p99 |
|---|---|---|---|
| `cnn_dual` (baseline) | **49.451** | 44.52 | 138.8 |
| `cnn_v2` | 49.273 | 44.64 | 138.9 |
| selected arm (it125) | 49.964 | 45.01 | 134.8 |
| matched control (it250) | 49.272 | 44.75 | 148.1 |

Versus `cnn_dual`:

| | mean delta [95% CI] | median delta | better |
|---|---|---|---|
| selected arm | **+0.514 [+0.17, +0.95]** | +0.216 | 848/2000 (z=−6.8) |
| matched control | −0.178 [−1.66, +0.86] | +0.033 | 971/2000 (z=−1.3) |
| **selected vs matched control** | +0.692 [−0.19, +2.11] | +0.154 | 882/1999 (**z=−5.3**) |

**Training on the 495 made the model worse**, by +0.514 with a CI that excludes zero, and it is worse
than training on 495 *random* segments from the same pool (sign test z=−5.3, though the mean CI on
that contrast includes zero). The matched control is indistinguishable from doing nothing.

## Why this was the expected outcome

The 495 were never `cnn_dual`'s weak set. The per-segment difference `cnn_dual − cnn_v2` has median
**−0.019** and a mean CI of [−0.28, +1.56] — the two nets are tied. 505/1000 is **z = +0.32** from a
fair coin. Conditioning on the winner therefore sorts segments by *difficulty*, not by affinity:
every controller in the table is better on the 505, including stock `pid` (−7.08) and `pid_lag`
(−9.08), which have no relationship to either net.

So "train on the 495" reduces to "train on the harder half, selected noisily" — which is
difficulty-specialisation, already tested and null in this project (`spec_train.py`, hard/all/rand
arms). It also shrinks the training set from 2000 segments to 495, on the half with the least
headroom. The fresh holdout confirms `cnn_v2` and `cnn_dual` remain tied: −0.178 [−3.09, +2.16],
better on 1003/2000.

## Process note

The first attempt at both runs was killed 32 minutes in by a machine reboot (uptime confirmed; not
an OOM — 5 of 59 GB in use), with no error written to either log. Both were restarted from scratch
rather than resumed from their last checkpoints, which sat at different iterations (100 vs 75) and
would have broken the matching. On restart both arms reproduced their iteration-0 validation values
exactly, confirming the seeding is deterministic.

`ablate.py` gained two options for this experiment: `TRAIN_LIST` (train on an arbitrary segment list)
and `TAG` (namespace the checkpoints). The `TAG` is not cosmetic — two runs of the same config
previously overwrote each other's checkpoints and destroyed a control arm.
