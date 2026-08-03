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

---

# Follow-up: the parameter is real, the gate took three attempts

## The parameter transfers

`gain_scale=2.1` was fit on the hard 25 of `ALL[5000:6000]`. Applied to the hard segments of the
**tuning** split, which it was never fit on:

| `gain_scale` | worst 25 (tuning) | saturating segments (n=32) | rest |
|---|---|---|---|
| 1.79 (shipped) | 245.27 | 178.89 | 43.89 |
| **2.10** | **193.43 (−21%)** | **142.09 (−21%)** | 47.86 |
| 2.50 | 277.62 | 206.15 | 54.76 |

Same shape, same optimum, on segments it never saw. The mechanism is real.

## Three gates, two failures

The whole difficulty is *when* to apply it. Tuning split, vs `ff_pi_boot` at 50.178:

| gate | best config | result |
|---|---|---|
| sigmoid on log `J*` (anticipatory, blunt) | gs 1.79, lam 8.0 | **+0.463** — every config worse |
| rate clamp binding, hold 3–8 (reactive) | gs 2.1, hold 8 | **+0.772** — every config worse |
| same, latching to end of segment | gs 2.1, hold 25 | **+1.383** — monotonically worse with hold |
| **peak abs tau over the preview (predictive, precise)** | **gs 2.1, tau 2.5–3.5** | **−1.917** |

The reactive gate is the informative failure. It is *precise* — it fired on 23 of 800 segments,
matching the saturating count almost exactly — and it still made those segments **worse**, better on
only 6 of the 23. By the time the clamp binds, the over-command has already happened; cutting a
memoryless feedforward afterwards only leaves the loop mismatched. This is why `pid_awu` worked and
this did not: the integrator is a *persistent state* that keeps growing during saturation, so freezing
it undoes ongoing damage. The feedforward has no memory, so there is nothing to undo — it has to be
prevented.

Peak `|tau|` over the preview separates the target far better than `J*` (3.78 vs 0.72) and fires
early enough to prevent rather than react.

## Validated result

| split | `ff_pi_boot` | `ff_pi_tau` | mean delta [95% CI] | fires on |
|---|---|---|---|---|
| tuning `ALL[3000:3800]`, n=800 | 50.178 | 48.262 | −1.917 [−4.82, −0.08] | 56/800 |
| **pristine `ALL[5000:8000]`, n=3000** | **53.248** | **50.742** | **−2.506 [−3.62, −1.54]** | 207/3000 |

p99 falls 253.9 → 173.8. The median delta is 0.000 — this is a pure tail fix, which is the honest
description of it. Promoted as the best classical controller.

## Process note

The identity gate caught a second silent bug here. `ff_pi_sat` initially reused the attribute names
`prev_lat` and `frozen`, which `ff_pi_boot` already uses for its inherited anti-windup. Writing
`prev_lat` before calling the parent made the parent compare the current lataccel to itself, so its
anti-windup never fired — worth +0.6 on the tuning split, and entirely invisible without the
"defaults must reproduce the base class bit-for-bit" check.
