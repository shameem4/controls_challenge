# Anti-windup: why the "easy" gains blow up, and how to keep them

## The observation that started it

Tuning `pid_boot` separately on easy and hard segments said both want `p=0.145`, and they differ only
in the integral gain: easy wants `i=0.140`, hard wants `i=0.100`. Applying the EASY gains everywhere
gives a strange result on pristine `ALL[5000:6000]`:

| | mean | median | better than nominal |
|---|---|---|---|
| `pid_boot` nominal (0.195, 0.100) | 69.455 | 55.69 | — |
| easy gains (0.145, 0.140) | **93.090** | **49.76** | 840/1000 |

It wins on 84% of segments and improves the median by 6 points, while the mean gets 24 points worse.

## Where the damage is

Not spread out. Concentrated in ten segments.

```
worst  10 segments contribute +21.14 of the +23.63 mean delta   (89.5%)
worst  25 segments contribute +27.69                            (117%; the remainder is net negative)
```

Instrumenting the plant's `MAX_ACC_DELTA` rate clamp and the integrator gives an unambiguous signature:

| group | n | mean delta | J* | sat. steps | max abs integ |
|---|---|---|---|---|---|
| worst 25 | 25 | +1107.46 | 35.85 | **29.0** | 15.60 |
| worst 26–160 | 135 | +12.08 | 2.90 | 0.0 | 4.41 |
| rest | 840 | −6.77 | 2.76 | 0.0 | 3.44 |

The failing segments spend 29 steps with the rate clamp binding; every other group spends **zero**. On
those same 25 segments the nominal gain saturates for only 8 steps, so the higher integral gain is what
drives the plant into the clamp. From there it is textbook windup: the clamp binds, the integrator
keeps accumulating against a limit it cannot move, the command overshoots, the overshoot re-saturates.
`pid_boot`'s `i_clip` defaults to `1e9`, so nothing bounds it.

## The fix, and what it is worth

Conditional anti-windup — freeze the integrator while the clamp binds, held for the plant's ~3-step
dead time. This is the same mechanism `ff_pi_rl2` already uses, and it was the largest classical win
in this project. `pid_boot` never had it.

Pristine `ALL[5000:6000]`, n=1000, versus nominal `pid_boot`:

| | mean delta | median delta | better |
|---|---|---|---|
| easy gains, no anti-windup | +23.634 | −5.93 | 840/1000 |
| easy gains **+ anti-windup** | **+3.431** | −4.54 | 846/1000 |
| nominal gains + anti-windup | −0.580 | 0.00 | 18/1000 |

86% of the tail damage removed, the median win kept, and it costs nothing at nominal gains because it
almost never fires there. A `hold`/`bleed` sweep on the separate tuning split `ALL[3000:3800]` put the
optimum at `hold=3` (the dead time) and `bleed=0`; every `bleed>0` was strictly worse. A hard `i_clip=5`
is **not** a substitute — it made things much worse (mean +45.97), because clipping still leaves the
integrator pinned at the limit while the clamp binds.

Anti-windup alone still does not beat nominal on the mean (+3.43). It needs the scheduler.

## `pid_fawu`: scheduled + reactive, stacked

`pid_fuzzy` and anti-windup attack the same failure from opposite sides:

* the scheduler is **anticipatory** — it backs the gain off wherever the preview's `J*` says the road
  is hard. That fires on hundreds of segments, most of which never actually saturate, so it gives up
  median performance broadly (median 53.31 against the easy gains' 49.76).
* anti-windup is **reactive** — it fires only where the clamp is genuinely binding (~25 segments), so
  it keeps the median, but leaves a positive tail.

Stacked, the scheduler can afford a much milder backoff (`i_hard` 0.100 → 0.120) because anti-windup
catches what it misses. Selected on the tuning split `ALL[3000:3800]`, then confirmed on pristine data:

**Pristine `ALL[5000:6000]`, n=1000, vs nominal `pid_boot`:**

| | mean | median | mean delta [95% CI] | median delta | better |
|---|---|---|---|---|---|
| `pid_boot` nominal | 69.455 | 55.69 | — | — | — |
| `pid_fuzzy` | 66.833 | 53.31 | −2.622 [−5.78, +0.12] | −1.743 | 745/1000 |
| **`pid_fawu` (i_hard=0.120, width=0.8)** | **66.124** | **50.53** | −3.332 [−7.06, +0.51] | −4.041 | 855/1000 |

**Fresh pristine `ALL[6000:8000]`, n=2000 — never touched by any tuning:**

| | mean | median | mean delta [95% CI] | median delta | better |
|---|---|---|---|---|---|
| `pid_boot` nominal | 68.478 | 55.16 | — | — | — |
| `pid_fuzzy` | 64.823 | 52.94 | — | — | — |
| **`pid_fawu`** | **64.835** | **50.87** | **−3.643 [−5.35, −1.95]** | **−3.798** | **1698/2000 (z=+31.2)** |

At n=2000 the mean CI **excludes zero**, which is the pre-registered bar `pid_fuzzy` failed. The point
estimate replicates across all three splits (−2.98 tuning, −3.33 pristine-1000, −3.64 pristine-2000).

Against `pid_fuzzy` directly on the fresh 2000: the means are indistinguishable (+0.012), but the
median is −1.346 better on 1557/1942 (z=+26.6). Anti-windup buys the median back without paying in
the tail — which is precisely the trade the scheduler alone could not make.

## What this does NOT do

`pid_boot` is not the deliverable. `ff_pi_boot` sits at 51.222 and `cnn_v2` at 46.911, so a 69 → 65
improvement to the PID family **does not move the headline**. This is a mechanism confirmation and a
better baseline, not a new best controller.

## The port to `ff_pi_boot` is a null

Scheduling `ff_pi_boot`'s gains the same way fails, and the reason is instructive. Tuning split
`ALL[3000:3800]`, n=800, vs `ff_pi_boot` (50.178):

| config | mean | mean delta | median delta | better |
|---|---|---|---|---|
| `ki_hard=0.12, w=1.6` | 50.050 | −0.128 | +0.471 | 241/792 |
| `ki_hard=0.10, w=0.8` | 53.684 | +3.506 | +1.484 | 153/785 |
| `ki_hard=0.08, w=0.8` | 60.659 | +10.481 | +3.235 | 100/787 |
| `ki_easy=0.16, w=0.8` | 52.400 | +2.221 | +0.924 | 223/800 |
| `ki_easy=0.19, w=0.8` | 53.356 | +3.178 | +1.237 | 253/800 |

Every departure from the tuned `ki=0.135` loses, in **both** directions. The best row's negative mean is
a tail artefact — its median is positive and it wins on only 30% of segments. Rejected.

Mechanism: `ff_pi_boot` already has all three things the schedule was supplying. It carries the
conditional anti-windup inherited from `ff_pi_rl2`, a bounded `i_clip=3.11`, and — most importantly —
cost-optimal Tikhonov smoothing of the reference itself. That smoothing already strips out the
hard-trajectory content that `J*` is measuring, so by the time the PI loop sees an error, the
difficulty the scheduler wants to react to has largely been removed upstream. Its tuned `ki=0.135` is
already near the pid side's *easy* optimum, which is the same fact seen from the other end.

## Process note

The first `ff_pi_fuzzy` cut hardcoded `kp=0.20, ki=0.10` from the base class's *docstring* signature
rather than inheriting the actual tuned attributes (`kp=0.142, ki=0.135`). The identity gate —
"parameters that should reduce to the base class must reproduce it bit-for-bit" — caught it
immediately: 55.485 against 50.178. Without that gate the null would have been reported at the wrong
magnitude and blamed on the schedule instead of on a silent gain overwrite.
