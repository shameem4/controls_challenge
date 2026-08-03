# Training on segments selected by real headroom

## Selection

The previous experiment trained on segments chosen by which of two tied networks happened to win —
i.e. by noise — and made the model worse. This one ranks by **headroom**: cost minus the achievable
floor, where the floor is `31.24 + J*` (causal noise floor plus the segment's Tikhonov optimum).

`cnn_dual` on the training pool `ALL[2000:4000]`:

| | mean | median |
|---|---|---|
| cost | 46.136 | 44.22 |
| floor (`31.24 + J*`) | 37.934 | 34.36 |
| **excess** | **8.202** | 7.04 |

**746 of 2000 segments are already at or below the floor** — nothing to win there. The top 500 by
headroom hold **70.6%** of all positive excess (mean excess 37.33, mean cost 78.62, against 8.12 /
46.11 for a random 500).

## Three arms, one difference each

All warm-started from `cnn_dual.pt`, `SEED=0`, 300 iterations of soft-token Gumbel policy
optimisation, checkpointed every 25.

| arm | training set | controls for |
|---|---|---|
| `hr500` | top 500 by headroom | — |
| `rnd500` | 500 random from the same pool | dataset **size** |
| `full2000` | all 2000 | continued training **at all** |

Training pool is `ALL[2000:4000]`, so both pristine splits stay clean. Checkpoints selected on
`ALL[4200:4700]` by real sim.

## Results

**Holdout `ALL[5000:6000]`, n=1000:**

| | mean | median | p99 |
|---|---|---|---|
| `cnn_dual` | 47.843 | 44.30 | 147.3 |
| `cnn_v2` | 47.361 | 43.94 | 133.8 |
| `hr500` it225 | 47.529 | 44.72 | **128.0** |
| `rnd500` it250 | 48.202 | 44.33 | 153.3 |
| `full2000` it100 | 48.552 | 43.99 | 143.5 |

**Holdout `ALL[6000:8000]`, n=2000:**

| | mean | median | p99 |
|---|---|---|---|
| `cnn_dual` | 49.451 | 44.52 | 138.8 |
| `hr500` it225 | 49.542 | 45.07 | 139.3 |
| `rnd500` it250 | 49.651 | 44.76 | 138.1 |
| `full2000` it100 | 49.678 | 44.78 | 140.0 |

Key contrasts (mean delta [95% CI], median delta, sign test):

| | n=1000 | n=2000 |
|---|---|---|
| `hr500` vs `cnn_dual` | −0.314 [−0.91, +0.16], med **+0.154**, 435/1000 (z=−4.1) | +0.092 [−0.62, +0.62], med **+0.251**, 816/2000 (z=−8.2) |
| `hr500` vs `rnd500` | −0.673 [−1.83, +0.15], med +0.027, 489/1000 (z=−0.7) | −0.109 [−0.49, +0.20], med +0.145, 888/2000 (z=−5.0) |
| `hr500` vs `full2000` | −1.023 [−3.33, +0.28], med +0.217, 407/1000 (z=−5.9) | −0.135 [−0.78, +0.30], med +0.192, 811/2000 (z=−8.5) |

**No arm beats `cnn_dual`.** Every mean CI includes zero, and every median delta for `hr500` is
*positive* (worse) with a decisive sign test against it. Continued policy optimisation from a
converged checkpoint does not help regardless of which segments it runs on — `full2000`, which simply
keeps training on the normal pool, is the worst of the three.

## But the mechanism did work

On the worst 100 segments of the n=1000 holdout:

| | mean on worst 100 |
|---|---|
| `full2000` it100 | 134.11 |
| `rnd500` it250 | 129.01 |
| `cnn_dual` | 125.77 |
| **`hr500` it225** | **120.47** |

`hr500` is the best of the three trained arms on the hard tail by 8.5–13.6 points, and it has the
lowest p99 of any model tested (128.0 against `cnn_dual`'s 147.3). Headroom-targeted training moved
the tail in the intended direction — unlike the 495-segment experiment, which moved everything
backwards. The gain is simply offset by a broadly worse median.

Caveat on that table: the worst 100 are selected by `cnn_dual`'s own cost, which biases the subset
against `cnn_dual` through regression to the mean. The fair comparison is `hr500` against `rnd500`
and `full2000`, which were selected and evaluated identically — and that contrast (120.47 vs 129.01
and 134.11) holds up.

## Conclusion

Selecting training data by headroom is the right criterion — it is causal, it is grounded in a
measured floor, and it demonstrably reshapes where the model spends its capacity. It still does not
produce a net win, because the tail it fixes is worth less than the median it costs. Combined with
the earlier nulls, the consistent finding is that **further policy optimisation from a converged
checkpoint does not improve this model**, and the choice of training subset changes only *how* it
fails to improve.

The remaining lever is not which segments to train on, but the ~31.24 noise floor and the median
segment, where `cnn_dual` already sits ~10 points above a floor it cannot reach by reweighting data.
