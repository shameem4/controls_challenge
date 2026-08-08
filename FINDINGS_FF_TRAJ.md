# Trajectory-error feedback on ff_pi: feedforward helps a lot, not enough

Follow-up to `FINDINGS_TRAJ_PID.md`. That result put trajectory-error control on stock `pid` and lost
6.3x. The objection: on stock `pid` the feedback loop carries the entire tracking burden, so any
phase it spends comes straight out of tracking. `ff_pi`'s inverse-plant feedforward supplies the
bulk of the command and costs no phase margin, so the trajectory loop would only correct residuals.

`controllers/ff_pi_traj.py`, on `ff_pi_boot`. Two changes from `pid_traj`: the trajectory error
accumulates against the SMOOTHED reference `c[k0]` the feedback term actually tracks, not the raw
target; and `H` defaults to 0 rather than assuming the lookahead helps.
**Identity gate (`w=0` reproduces `ff_pi_boot`): PASS.**

## The objection was correct in direction

Pure trajectory control costs 4.5x baseline here (185.71 vs 41.64) against 6.3x on stock `pid`
(485.37 vs 77.50). At low blend weights the two are within noise of each other. Feedforward
genuinely does absorb most of the phase penalty.

## It still never crosses over

Held out, `ALL[5000:5300]`, n=300, paired against the parent:

| arm | total | lat | jerk | paired delta | 95% CI | better on |
|---|---|---|---|---|---|---|
| `ff_pi_boot` (w=0) | **57.833** | 0.7318 | 21.243 | -- | -- | -- |
| w=0.10 | 58.444 | 0.7602 | 20.433 | +0.611 | [-0.224, +1.321] | 74/300 |
| w=0.25 | 61.283 | 0.8364 | 19.463 | +3.450 | [+1.939, +4.850] | 28/300 |

`w=0.10` is not separable from baseline in the mean, but wins on only 74/300 segments -- worse by
sign test, not an improvement. `w=0.25` is significantly worse.

## The number that matters

The trade is jerk bought with tracking. In cost units:

| parent | d(jerk) | d(lat) | exchange rate | jerk weight needed to break even |
|---|---|---|---|---|
| stock `pid`, w=0.15 | -4.62 | +10.95 | 0.42 | 2.37x current |
| `ff_pi_boot`, w=0.10 | -0.810 | +1.421 | **0.57** | **1.75x current** |

Feedforward moved the crossover from about 2.4x the current jerk weight to 1.75x. Real progress on
the mechanism; the cost function still does not pay for it.

> **Corrected 2026-08-08 after a bug bash.** This table first read `+11.20` / `0.41` / `~7x` for the
> `pid` row. The d(lat) figure was arithmetically wrong -- `50*(2.199-1.980)` is 10.95, not 11.20 --
> and, more seriously, the `~7x` break-even did not come from this row at all: it was carried over
> from the 60-segment TUNING set (where the same computation gives ~5.6x) and presented alongside
> held-out numbers. Break-even implied by this row's own figures is `10.95 / 4.62` = 2.37x. The
> claim "feedforward moved the crossover from roughly 7x to 1.75x" was therefore overstated; the
> honest version is 2.4x to 1.75x. The `ff_pi` row was correct as published.
>
> Caveat that stands: the two rows come from different held-out samples (n=200 for `pid`,
> `ALL[5000:5200]`; n=300 for `ff_pi_boot`, `ALL[5000:5300]`), so the comparison between them is
> indicative rather than matched.

## Two null results worth recording

**Gain shape is inert.** Sweeping `k_psi` over 8x (0.05 to 0.40) and `tau` over 3x (1.0 to 3.0)
moved the total from 42.47 to 45.82, with no interior optimum. Only the blend weight `w` matters --
the trajectory term's magnitude counts, its shape does not.

**Lookahead is null.** `H` = 0, 3, 8, 15 gave 42.47 / 42.48 / 42.44 / 42.29 at w=0.1, and at w=0.25
larger `H` was slightly WORSE. This is the third time added anticipation has come back null on top
of `ff_pi_rl2`: the smoothed-reference feedforward already anticipates, and `FINDINGS_PREVIEW.md`
showed the cost can only use about 1 s of preview regardless. Note the request that motivated this
arm was specifically that "trajectory benefits from feedforward/lookahead" -- the feedforward half
was right and measurable, the lookahead half was not.

Not promoted. `master` untouched; work retained on the `traj` branch.
