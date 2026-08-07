# The feedforward gain model was wrong, and `gain_scale` was hiding it

## What was measured

`verify_plant.py` steps the action from equilibrium and reads the settled lataccel in `expected`
mode -- the plant's conditional mean, so there is no sampling noise and a single run suffices.

| v (m/s) | 8 | 15 | 22 | 28 | 34 |
|---|---|---|---|---|---|
| measured DC gain | 2.445 | 2.533 | 2.243 | 2.424 | 2.315 |
| `gain_fit.npy` G(v) | 1.422 | 1.421 | 1.475 | 1.563 | 1.692 |
| ratio | 1.72 | 1.78 | 1.52 | 1.55 | 1.37 |

Two independent errors:

**Magnitude.** `gain_fit.npy` is low by roughly 1.6x. `ff_pi`'s tuned `gain_scale = 1.79` has been
silently absorbing exactly that -- it was correcting a systematic identification error, not detuning
the loop as its name implies. This is why `gain_scale` needed tuning at all, and it retires an
Unknown that had been carried without explanation.

**Slope.** The measured gain is essentially FLAT across 8-34 m/s while `gain_fit.npy` rises, so the
ratio falls monotonically. The speed dependence we modelled is not there. That is consistent with
`FINDINGS_SYSID.md` ("gain depends on operating point, not just speed") -- and the operating-point
dependence is already handled by `ff_pi_tau`'s tau-gated detune.

An outside solution (`nurikserikbayev`) independently reports ~2.0 from excitation-based ARX
identification. Their number is much closer to the truth than ours was.

## Does fixing it help?

`ff_pi_flat` replaces `G(v)` with a constant, applied multiplicatively through `gain_scale/gs_norm`
so the tau modulation survives. `flat=None` reproduces `ff_pi_tau` bit-for-bit.

Pristine `ALL[5000:8000]`, n=3000:

| | mean | median | p99 |
|---|---|---|---|
| `ff_pi_tau` | **50.742** | 45.57 | 173.8 |
| flat 2.5, gs_hard 2.1 | 50.877 | 45.14 | 216.7 |
| flat 2.3, gs_hard 2.5 | 50.574 | **44.79** | 190.8 |

| | mean delta [95% CI] | median delta | better |
|---|---|---|---|
| flat(2.5, 2.1) | +0.135 [−0.40, +0.91] | −0.337 | 2022/2956 (z=+20.0) |
| flat(2.3, 2.5) | −0.168 [−0.68, +0.40] | −0.707 | 2129/2999 (z=+23.0) |

**Mixed, and reported as such.** On the headline metric (mean) it is a wash -- both CIs span zero,
and the tuning-split mean advantage (49.05 vs 49.87) did NOT replicate. On the median and the
per-segment win rate it is a clear, decisive improvement: 71% of segments better, sign test z=+23.

Not promoted, because the deliverable is scored on the mean. But the finding stands on its own: the
gain model is wrong in two measurable ways, and correcting it improves the typical segment for free.

## The interesting part

That correcting the gain to the measured truth does **not** improve the mean is itself informative.
It says the tuned `gain_scale` was already an adequate average correction, and what the fix buys is
in the bulk rather than the tail. The tail is governed by the operating-point dependence, which is
what the tau gate handles -- and the best flat configuration wants a *stronger* tau detune
(`gs_hard` 2.5 vs 2.1), which is what one would predict if the flat model is right about speed and
the remaining variation is operating-point.

## Process note

The first version of this test overrode `G()` to return a bare constant, which silently disabled the
tau-gated detune (`ff_pi_tau` modulates `gain_scale`, and the override ignored it). That made flat
gain look worse than sloped (51.6 vs 49.9) when in fact it was being compared with a ~1.75-point
mechanism switched off. Applying the constant multiplicatively through `gain_scale/gs_norm` fixed it.
