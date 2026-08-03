# Segment-routed portfolio: the oracle ceiling, and how much of it is real

Built what was asked: `controllers/portfolio.py` dispatches a different strategy per segment from a
lookup table keyed on segment identity. Verified end-to-end — the routed controller reproduces the
oracle computed from the cost matrix **exactly** (max abs difference 0.00e+00 over 1000 segments).

A table fit on the segments it is evaluated on is an oracle, not a controller — the same category as
the sub-30 leaderboard exploits. It is useful only as an upper bound on what a causal router could buy.

## The raw ceilings look encouraging

Pristine `ALL[5000:6000]`, n=1000. Single arms:

| arm | mean | median |
|---|---|---|
| `pid_boot` nominal | 69.455 | 55.69 |
| `pid_boot` easy gains | 93.090 | 49.76 |
| `pid_awu` easy | 72.886 | 49.59 |
| `pid_fuzzy` | 66.833 | 53.31 |
| `pid_fawu` | 66.124 | 50.53 |
| `ff_pi_boot` | 52.930 | 45.23 |
| `cnn_v2` | 47.361 | 43.94 |

Oracle routing over subsets:

| portfolio | oracle mean | best single arm | gap |
|---|---|---|---|
| {`pid_boot` nominal, `pid_awu` easy} | 61.553 | 69.455 | **+7.90** |
| all pid variants | 58.637 | 66.124 | +7.49 |
| {`pid_fawu`, `ff_pi_boot`} | 52.050 | 52.930 | +0.88 |
| {`ff_pi_boot`, `cnn_v2`} | 45.489 | 47.361 | +1.87 |
| everything (8 arms) | 45.287 | 47.361 | +2.07 |

## Most of that gap is winner's curse

Taking a per-segment minimum over noisy arms flatters itself. The control is to oracle-route between
arms that are **equally good but genuinely different**, where by construction there is nothing to
route on. Any gap there is pure selection noise.

A first attempt at this null was too weak: `p=0.195` vs `p=0.1951` gave a gap of only +0.06, but the
median per-segment cost difference between those two arms is literally **0.000** — the plant's output
tokens are discrete, so a 0.05% gain change usually produces an identical trajectory. Two arms that
never differ cannot generate selection noise.

Nearby PID settings of comparable quality (`dQ` = difference in mean cost) do differ, and they show:

| pair | dQ | oracle | **null gap** |
|---|---|---|---|
| (0.215, 0.095) vs (0.235, 0.090) | 1.52 | 69.014 | +1.010 |
| (0.195, 0.100) vs (0.215, 0.095) | 0.57 | 67.883 | +1.573 |
| (0.195, 0.100) vs (0.175, 0.105) | 1.03 | 66.622 | +1.799 |
| (0.175, 0.105) vs (0.155, 0.110) | 1.50 | 66.444 | +1.978 |
| (0.195, 0.100) vs (0.235, 0.090) | 2.09 | 67.329 | +2.127 |
| (0.155, 0.110) vs (0.215, 0.095) | 0.11 | 65.588 | +4.328 |
| (0.155, 0.110) vs (0.235, 0.090) | 1.63 | 65.170 | +4.747 |

**Routing between two arbitrary nearby PID tunings "buys" 1.0–4.7 points of nothing.** So of the
+7.90 headline gap, a substantial fraction — plausibly a third to over half — is noise.

## At the top of the table, the complementarity is exactly zero

The same test run on the controllers that actually matter:

| pair | oracle | gap | |
|---|---|---|---|
| `ff_pi_boot` lead=2 vs `ff_pi_boot` lead=3 | 50.870 | +1.856 | **same-family (null)** |
| `ff_pi_boot` lead=3 vs `ff_pi_boot` lam=4.0 | 50.832 | +1.893 | **same-family (null)** |
| `ff_pi_boot` lead=2 vs `cnn_v2` | 45.489 | **+1.872** | cross-family |
| `ff_pi_boot` lead=3 vs `cnn_v2` | 45.473 | **+1.888** | cross-family |
| `ff_pi_boot` lam=4.0 vs `cnn_v2` | 45.533 | +1.828 | cross-family |

The cross-family gap between a classical controller and a learned one (**+1.87**) is
**indistinguishable from the gap between two copies of `ff_pi_boot` that differ only in their preview
lead** (+1.86, +1.89). A learned CNN and a hand-tuned feedforward-PI are, as far as this test can
tell, *not complementary at all*. Their apparent per-segment disagreement is the same magnitude of
noise you get from perturbing one controller's lookahead by a single step.

## What is causally reachable

For the pair with the largest raw gap, {`pid_boot` nominal, `pid_awu` easy}, using the causal
difficulty signal `log J*` computed from the preview:

* `pid_awu easy` wins on 846/1000; oracle gap **+7.90**
* AUC of `log J*` for predicting which arm wins: **0.578** (0.5 = no signal) — weak
* 2-fold cross-validated threshold router on `log J*`: **65.643** mean, recovering **48.2%** of the gap

That 48.2% is measured out-of-fold, so it is real. It also beats the best single pid arm (68.42) by
~2.8 points and marginally beats `pid_fawu`'s soft blend (66.124). Routing inside the PID family
works.

## Conclusion

Routing is real where the arms are genuinely different in *kind* and both are weak — inside the PID
family it recovers ~3.8 points out-of-fold. It is worth nothing where it matters: the entire PID
family sits ~18 points behind `cnn_v2`, and between `ff_pi_boot` and `cnn_v2` the measurable
complementarity is zero.

The "see-saw" is therefore not a routing problem. Two controllers trading wins segment-by-segment at
the same average cost is the expected signature of two similar controllers being hit by different
noise draws, not evidence of specialisation waiting to be exploited. The null-oracle control above is
the cheap way to tell those apart, and should be run before any future portfolio or
mixture-of-experts result is believed — it would have pre-emptively killed the MoE, the ensemble, and
the difficulty-specialist experiments, all three of which were nulls.

The remaining headroom is in making a single controller better, not in choosing between existing ones.
