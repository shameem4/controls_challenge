# Where `ff_pi_boot` is actually difficult

## Defining "difficult" correctly

Raw cost is the wrong ranking — some segments are expensive no matter what. Two floors must be
subtracted before a segment counts as a controller failure:

* the **trajectory floor** `J*`, the closed-form Tikhonov optimum on that segment's target;
* the **causal noise floor 31.24**, derived in `FINDINGS_FLOOR.md` and irreducible for *any* controller.

Using `J*` alone — the obvious mistake, and the one I made first — reports a median "excess" of 39.66
against a median cost of 45.23, i.e. it claims almost the entire cost is the controller's fault. That
is wrong by roughly the noise floor. With both floors:

| | mean | median |
|---|---|---|
| `ff_pi_boot` cost | 52.93 | 45.23 |
| approx achievable (31.24 + `J*`) | 38.31 | 34.14 |
| **controller excess** | **14.62** | **8.42** |

**354 of 1000 segments are already at or below the approximate floor.** There is nothing to win there.

## The excess is concentrated

| | share of total controller excess |
|---|---|
| worst 25 | 51.7% |
| worst 100 | 74.9% |
| worst 250 | 101.9% (the remainder is net negative) |

`ff_pi_boot` has no broad tuning deficit. It has a tail.

## The actionable target: 25 segments

Comparing per-segment against `cnn_v2`, whose 5.57-point mean advantage is the gap actually worth
closing:

| group | n | `ff_pi_boot` | `cnn_v2` | gap | `J*` | sat. steps | max abs tau | v (m/s) |
|---|---|---|---|---|---|---|---|---|
| worst 25 | 25 | 355.83 | 151.59 | **+204.24** | 49.72 | 6.9 | 3.78 | **14.1** |
| 26–100 | 75 | 67.05 | 51.98 | +15.06 | 16.47 | 0.6 | 1.60 | 18.8 |
| 101–300 | 200 | 46.35 | 42.16 | +4.19 | 7.44 | 0.1 | 0.93 | 22.8 |
| rest | 700 | 42.48 | 44.63 | **−2.15** | 4.43 | 0.1 | 0.72 | 24.3 |

**25 segments carry 91.7% of the entire gap to `cnn_v2`.** On the other 700, `ff_pi_boot` is *better*
than the CNN. Their signature is unambiguous and causally observable from the preview: high `J*`
(49.7 vs 4.4), large peak lateral acceleration (3.78 vs 0.72), and **low speed** (14.1 vs 24.3) —
tight, slow, hard corners. 30% of the worst 100 hit the plant's rate clamp, against 2% elsewhere.

## Mechanism: the feedforward over-drives at low speed

`ff_pi_boot` inverts the plant with `ff = (c_lead − roll) / G(v)`, where `G(v) = gain_scale · poly(v)`.
Sweeping `gain_scale` on the hard 25 versus the other 975 (diagnostic only — 25 segments, selected on
this same data, so the magnitudes are optimistic):

| `gain_scale` | hard 25 | rest 975 |
|---|---|---|
| 1.30 | 545.47 | 46.62 |
| 1.60 | 382.30 | 45.18 |
| **1.79 (shipped)** | **355.83** | **45.16** |
| **2.10** | **221.82** | 48.52 |
| 2.50 | 315.17 | 54.51 |
| 3.20 | 492.64 | 58.35 |

A *larger* `gain_scale` means a *smaller* feedforward command. Weakening it cuts the hard-25 cost by
**38%** for +3.4 on the rest. So on these segments the inversion is commanding too much, driving the
plant into its rate clamp — which is exactly what the saturation counts show.

This connects directly to the earlier system-ID result (`6723bd8`, "gain depends on operating point,
not just speed"): the plant's true gain is higher at large lateral acceleration than the speed
polynomial predicts, so `1/G(v)` over-commands precisely where `|tau|` is large.

A second, independent lever: reference smoothing `lam` = 8.0 gives hard-25 305.73 for rest-975 45.86,
against the shipped 355.83 / 45.16 — a 14% improvement at almost no cost elsewhere.

Structural ablations confirm nothing else is broken: pure feedforward (1075.62), P-only (687.50),
`smooth=False` (530.33) and `i_clip=1.0` (356.84 hard / 132.80 rest) are all far worse. The
architecture is right; the operating-point gain is not.

## Important caution

This is **not** the |lataccel| gain schedule that was already tested and rejected in both directions.
That one scheduled the *feedback* gains. This is the *feedforward inversion* gain, a different
quantity with a measured physical justification.

The `gain_scale=2.1` number is fit on the same 25 segments it is measured on and must not be believed
as a result. The honest test is to define the hard set from a **causal** preview signal (peak `|tau|`
and `v_ego` over the lookahead, both available at every step), schedule `gain_scale` on it, and
evaluate on segments never used for selection. That is the next experiment, and unlike the previous
scheduling nulls it starts from a measured mechanism rather than a hope.

Reproduce: `python diag_ffpi.py 5000 1000`. Segment list: `ffpi_hard_segments.csv`.
