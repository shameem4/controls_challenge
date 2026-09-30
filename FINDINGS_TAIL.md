# Where the classical controller actually loses

*Recorded on branch `tail-analysis`, last updated 2026-07-28. Headline numbers quoted here are those current at the time; `master` has since moved on — see the README for the figures that stand today.*

An investigation into the gap between `ff_pi_tuned` (55.68) and `cnn` (48.14) on the clean
500-segment split `[500:1000]`. It is mostly a record of **negative results** — five mechanism
hypotheses died on measurement — plus one live structural lead and one methodological warning.

Everything here is reproducible from `analyze_hard.py`, `ortho_sched.py` and `trace_tail.py`.

---

## 1. The gap is six segments, not a broad deficit

Per-segment costs for `ff_pi_tuned` over `[500:1000]`:

| | share of total cost | mean excluding them |
|---|---|---|
| worst 1% (5 segs) | 12.1% | 49.46 |
| worst 5% (25 segs) | 22.0% | 45.71 |
| worst 10% (50 segs) | 30.4% | 43.09 |

mean 55.68, median 46.79, **max 1994.8**.

Against `cnn` on the six worst:

```
 seg      v  tgt_max |      J*   ff_pi_tuned      cnn      pid
 133   17.1     4.19 |    71.6        1994.8    232.1   1947.9
 348   10.7     3.70 |    37.1         466.3     87.0    388.1
 265   12.9     3.36 |    40.3         367.3     82.5   1345.6
  22    7.1     3.06 |   100.5         284.1    282.3    813.9
 222   17.6     3.91 |    30.8         249.3     71.8    773.6
 195   28.7     3.05 |    36.6         213.6     96.4    418.0
mean                 |                 595.9    142.0    947.9
```

The gap on these six is 453.9 each; spread over 500 segments that is **5.45 of the 7.54-point
total gap — 72%**. `cnn` is not broadly better; it is better at not blowing up. (Seg 22 is the
exception: `cnn` ties at 282 and `J*` is the highest at 100.5, so that one is genuinely hard.)

These segments are **low speed with large lateral acceleration** — tight cornering. They are not
infeasible: `cnn` cuts them 4.2×.

## 2. Speed-regime gain scheduling is capped — with a hard number

The intuition that stratifying by speed would tame the variance and permit per-regime tuning does
not survive measurement. Cost σ within speed quintiles versus overall:

```
overall sigma = 95.34          break-even for 5 bins: <= 42.63 (0.447x)
  bin 1 v= 1.5-15.5  mean= 44.61  sigma= 71.89  (0.75x)
  bin 2 v=15.5-23.7  mean= 60.49  sigma=197.60  (2.07x)
  bin 3 v=23.7-28.3  mean= 49.73  sigma= 18.57  (0.19x)
  bin 4 v=28.3-31.2  mean= 58.12  sigma= 26.35  (0.28x)
  bin 5 v=31.2-38.4  mean= 65.48  sigma= 16.57  (0.17x)
pooled within-bin sigma = 95.42 = 1.00x overall   ->  SE per bin 2.24x worse
```

**Speed-stratifying reduces cost variance by exactly zero.** Speed shifts the mean cleanly
(44.6→65.5) but the variance is a heavy tail concentrated at low-to-mid speed; high-speed segments
are extremely homogeneous (0.17–0.28×). So a 5-regime lookup table pays the full √5 standard-error
penalty for no variance reduction.

Binning also faces a resolution/data conflict with no clean resolution: with equal-width bins the
0–8 m/s cell holds 24 of 400 tune segments (and 60 is already known to overfit badly), while with
equal-population bins the lowest cell spans 1.9–15.1 m/s — too wide to resolve a detuning ramp that
moves +36%→+24% between v=4 and v=10. Parametric forms pool all segments across shared parameters
and sidestep this; that is what `ff_pi_sched` does.

What is safe is **parameters per bin, not bins**: 1 free number per cell plus shared globals keeps
the same params-per-segment density as the working global tune. 5 bins × full param set does not.

## 3. Optimal feedforward is a *detuned* plant inverse — and the detuning shape matters

Directly probed plant gains versus what the tuned controllers use (larger `G` ⇒ less steer):

```
  v     truth   ff_pi_tuned  err%  |  ff_pi_probed  err%
  4      1.91         2.59   +36   |         2.23   +17
 10      2.05         2.54   +24   |         2.40   +17
 22      2.39         2.64   +10   |         2.79   +17
 34      2.80         3.03    +8   |         3.26   +17
```

Both **over**-estimate `G`, i.e. deliberately under-steer relative to true plant inversion. But the
optimal detuning *falls with speed* (+36%→+8%), and a single `gain_scale` on the true curve gives a
constant relative detuning, so it structurally cannot express that ramp. Result: `ff_pi_probed`,
using the **correct** gain curve, scored 58.02 against `ff_pi_tuned`'s 53.40 on `[500:620]`, with
**69% of the deficit in the lowest speed tercile**. The original wrongly-flat curve, scaled up,
happened to approximate the right detuning shape: wrong as physics, right as a controller.

At low speed the conservative controller is better on **both** terms (lat 0.59 vs 0.72, jerk 14.23
vs 17.17), so this is not a tracking/smoothness trade — extra feedforward authority there is
actively harmful. Why the required detuning is ~4× larger at low speed is **unexplained**.

> **Caveat added after §7.** The detuning-*shape* explanation above is weak. Scheduling the
> detuning explicitly (`ff_pi_sched`, §7) was fitted to the measured +36%→+8% profile and still
> landed at ~58, the same as the flat-scalar `ff_pi_probed`. Every variant built on the probed
> curve scores ~58 while the original curve scores 53.40, so **the common factor is the curve, not
> the schedule**. See §7 for the likely reason.

## 4. Five dead hypotheses for the tail

Each was plausible, and each was killed by direct measurement rather than argument.

| hypothesis | test | verdict |
|---|---|---|
| **actuator saturation** | `\|a\|max` on the 6 worst = 1.86, 1.76, 1.35, 1.97, 1.37 — only 1 of 6 saturates | dead. `act_sat`'s +0.902 correlation with excess is a **single-outlier artifact** |
| **λ over-smoothing** on large-amplitude targets | λ sweep on 40 worst: 430 → 575 → 650 → 596 → 574 → 548 for λ=1…8 | dead — **non-monotonic**; a real amplitude mechanism would be monotone. Typical segments vary only 1% across λ∈[1,8], so **λ is a nearly flat direction and its tuned value is largely noise** |
| **oscillation / instability** | 0 of 40 flag. Tail flips sign *less* than typical (1.56 vs 2.07/s); `dact_ac1` **positive** (+0.85) = smooth control; `f_peak` 0.26 Hz ≈ typical | dead |
| **divergence / offset** | `err_drift` 0.023 vs 0.021; `err_bias` −0.004 vs −0.003 | dead |
| **integrator windup** against `i_clip` | `integ_pin` is 7.6× higher in the tail (7.3% vs 1.0%), but raising `i_clip` makes the tail monotonically **worse**: 189.9 → 260.7 → 346.9 → 475.8 for 3.11 → 5 → 10 → 100 | dead — the clamp is *protecting* us; pinning is a symptom. Note typical segments are bit-identical for `i_clip` ≥ 3.11, so **`i_clip` was tuned entirely by the tail** |

Consequence: every `ff_pi` parameter we have swept — λ, `i_clip`, `lead`, the gain curve and its
detuning — is already at or near its optimum. **The tail failure is structural, not parametric.**

## 5. The live lead: a static-gain feedforward cannot cancel plant lag

Decomposing burst error, `lat − tgt = (lat − desired) + (desired − tgt)`, over the 8 worst segments:

```
|lat-ref| = 0.437    |ref-tgt| = 0.120    reference share of burst error = 14.3%
|dtgt| during bursts = 4.45x quiet (up to 14x)
```

So **86% of burst error is the plant lagging its own reference**, not the smoother abandoning the
target — and bursts are exactly the fast-target-movement moments. Cross-correlating achieved
against reference during bursts:

```
mean best lag = 2.12 steps = 213 ms      burst amplitude ratio = 1.119
```

Pure **phase lag** at essentially unit amplitude (an amplitude-limited loop would show ratio ≪1).
That is the signature of an uncancelled first-order plant lag. `ff_pi`'s feedforward is
`(c[t+lead] − roll) / G(v)` — a **static gain inverse**, which cannot cancel dynamics at any gain.
`lead` is only a crude integer-delay hack, and its ceiling confirms this:

```
 lead   40 worst   40 typical
    2      189.9        46.86  <- tuned
    3      185.0        47.38
    4      185.7        49.42
    6      228.0        54.00
```

Best case is 0.39 points of mean cost — negligible against the 7.5-point gap.

**Proposed next step:** a genuinely dynamic feedforward — invert a first-order/ARX plant model
(lead-lag compensator, textbook 2-DOF design) instead of a static gain. The LPV-ARX models fitted
in `lpv_id.py` on the `mpc-lpv` branch already exist for this. This is also plausibly what `cnn`
learned implicitly by training through the differentiable plant, which would explain a 4.2× tail
advantage without broad superiority. **Not yet tested.**

## 6. Methodological warning

Two of the dead hypotheses above were generated by *me* from aggregate statistics over 400
segments, asserted, and then refuted within minutes. On a distribution with median 46.8 and max
1994.8, correlations and OLS R² are dominated by one or two segments:

- `act_sat` showed raw corr **+0.902** with excess — entirely one segment.
- Speed-only R² on excess came out at **0.004**, apparently contradicting the well-supported
  finding that difficulty tracks speed (`v_min` differs 4.15× between best and worst excess
  quintiles). Both are computed correctly; quintile means are robust to the tail, R² is not.

Corollary: `ortho_sched.py`'s printed verdict ("2-D schedule justified — `tgt_rms` carries
orthogonal structure") is **not trustworthy** for the same reason, and should be recomputed on
ranks or with the tail winsorised before anyone acts on it. On this metric, use robust statistics
or per-segment traces; aggregate correlations are not evidence.

## 7. Status

### Speed-scheduled detuning: a sixth negative result

`ff_pi_sched` (probed curve + 3-parameter detuning schedule `d(v) = d_hi + (d_lo−d_hi)e^{−v/vc}`,
8 free parameters, CMA-ES on 400 segments) **overfits**:

```
tune set          50.59 -> 47.19        (apparent 3.4-point gain)
held-out [500:620] 58.04 -> 58.95        (+0.91 WORSE)
```

The held-out guard in `tune_cma.py` fired. Eight parameters on 400 heavy-tailed segments is past
what this objective supports — the same params-per-segment argument as §2, now confirmed for a
parametric schedule and not just for bins. The search also drove `d_hi` **negative** (−0.079),
i.e. asking to *over*-steer at high speed, contradicting the +8% detuning measured from
`ff_pi_tuned`; that sign flip was visible mid-run and is a marker of a weakly-identified direction.

**The whole speed-scheduling thread is closed.** Held-out comparison on `[500:620]`:

| controller | gain basis | held-out |
|---|---|---|
| `ff_pi_tuned` | original curve + scalar | **53.40** |
| `ff_pi_probed` | probed curve + scalar | 58.02 |
| `ff_pi_sched` | probed curve + 3-param schedule | 58.95 (58.04 at defaults) |

Every variant built on the probed curve lands at ~58 regardless of how the detuning is scheduled,
while the original curve reaches 53.40. **The common factor is the gain curve, not the schedule** —
which is why §3's detuning-shape story does not hold up.

**Likely reason (hypothesis, untested).** The probe measured a *static* gain, but §5 established the
plant has ~213 ms of lag. A dynamic plant has no single correct static gain: its DC gain exceeds its
gain at the frequencies where the target actually moves, so inverting the DC gain is the wrong
operation, and it under-steers precisely on fast transients. The original `gain_fit.npy` was fitted
from trajectory regression rather than steady-state probing, so it may have captured an effective
mid-frequency gain — better as a controller while worse as physics. If true, §3 and §5 have a single
cause and a single fix.

## 8. Dynamic feedforward: seventh negative — and the plant is dead-time, not lag

`ff_pi_dyn` inverts a first-order model along the reference,
`u = (y*[t] − a·y*[t−1]) / (G(1−a))`, which has DC gain exactly `1/G` (so steady state is
unchanged) and supplies phase advance only where the reference moves. `pole=0` reproduces `ff_pi`
bit-for-bit. Sweep on 40 worst / 40 typical:

```
  pole  lag(steps)   40 worst   40 typical
  0.00        0.00      189.9        46.86  <- current
  0.20        0.25      180.9        46.72
  0.35        0.54      192.8        46.83
  0.68        2.13      206.2        48.14  <- the value matching the measured 213 ms
  0.82        4.56      266.7        53.57
```

At the theoretically indicated pole it is **worse**. The small dip at `pole=0.2` was tested properly
on `ALL[1000:1500]` (500 segments never used for anything else):

```
pole 0.00 mean=56.617    pole 0.20 mean=56.342
paired delta = -0.275 +- 0.698 (SE)  t=-0.39   improved 239/500, worsened 253/500
-> indistinguishable from noise
```

**Why it was structurally wrong.** Direct small-signal step response of the plant (`gain_id.py`,
256 segments, `expected` mode, baseline = a nominal *tracking* trajectory, since the dataset's
`steerCommand` is NaN past step 100):

```
  k      0      1      2      3      4      5      6      7      8
  s/G -0.011  0.008  0.091  0.315  0.695  0.991  0.990  0.999  0.997
  dead time (>10% of DC) = 3 steps = 300 ms;  fully arrived by k=5
```

The plant is **~3 steps of dead time followed by a fast rise**, not a first-order lag — a
first-order lag has its largest increment at k=1, where this has 0.008. **Dead time has no causal
inverse**, so a lead-lag feedforward cannot cancel it at any pole; delay is compensated by
*prediction*. That is exactly what `lead` does, and `lead`'s tuned optimum of 2–3 steps already
matches the measured delay. `ff_pi` was already compensating the plant's delay correctly, which is
why both the `lead` and `pole` sweeps found nothing to take.

My §5 inference was wrong in the model, not the number: 213 ms measured in *closed loop* was
attributed to a first-order *plant* lag; it is transport delay, already compensated.

### The gain curve, settled

Same experiment, `G` by speed, with linearity verified across `du ∈ {0.05, 0.1, 0.2}` and
`tstep ∈ {140, 220, 300}` (spread 1.561–1.699, 9% — static gain is a valid picture):

```
    v band    n  G_meas  gain_fit  probed  fit err%  probed err%
  1.2-14.9   51   1.301     1.417   2.042       +9         +57
 14.9-22.7   51   1.516     1.442   2.289       -5         +51
 22.7-28.1   51   1.676     1.531   2.522       -9         +50
 28.1-31.6   51   1.751     1.598   2.648       -9         +51
 31.6-37.4   52   1.825     1.675   2.772       -8         +52
```

**`gain_fit.npy` is accurate to ±9% in magnitude and slope. `gain_fit_probed.npy` overstates by
~50% uniformly and is refuted.** That closes §7 on cause rather than outcome: `ff_pi_probed` and
`ff_pi_sched` were inverting a gain half again too large. `gain_fit_probed.npy` should not be used;
`gain_fit_step.npy` is the measured replacement, though `gain_fit.npy` is already within tolerance.

(A first pass at this used a `u=0` baseline on 64 segments and reported `G` *decreasing* with
speed. That was off the operating manifold and is superseded by the on-manifold result above.)

**The detuning is much larger than §3 claimed.** Against measured truth:

```
v= 9.6  truth=1.301  ff_pi_tuned=2.537  detune=+95%
v=18.6  truth=1.516  ff_pi_tuned=2.582  detune=+70%
v=33.3  truth=1.825  ff_pi_tuned=2.999  detune=+64%
```

`ff_pi_tuned` commands roughly **half** the steer true plant inversion calls for. §3's "+36%→+8%"
figures were computed against the probed curve and are void. The 300 ms dead time explains *why*:
delay caps stable loop gain, so heavy detuning is forced. **Dead time is the single cause behind
both the detuning and the tail.**

Caveat that closes the follow-up: those percentages are derived *from* `ff_pi_tuned`, so "optimal
detuning" is circular — it is what CMA-ES landed on, not an independent optimum. Rebuilding a
schedule on the corrected curve would reproduce `ff_pi_tuned`. No gain remains in the gain curve.

## 9. Why cnn wins, mechanically

The plant has ~300 ms of irreducible dead time. `ff_pi` compensates it with a single tap on a
Tikhonov-smoothed reference, `c[t+lead]` — a weak predictor when the target is moving 4.45× faster
than usual, which is precisely the burst condition (§5). `cnn` consumes 25 steps of preview through
a learned nonlinear map: a better predictor over the delay horizon. Its advantage is not better
tracking everywhere (§1: six segments carry 72% of the gap) but better prediction where prediction
is the binding constraint.

This also explains why every global reparameterisation of `ff_pi` failed. Optimal behaviour is
globally conservative (detune ~2×, forced by dead time) but locally aggressive on fast transients.
A fixed linear controller cannot be both; a nonlinear one conditioned on preview can. **The
limitation is that `ff_pi`'s aggression is state-independent** — not any particular parameter.

## 10. State-dependent authority: eighth negative, and §9 refuted by its own experiment

`ff_pi_adapt` scales the feedforward by `1 + beta·min(r/r0, 1)` where `r = |dc/dt|` at the tap, so
authority rises (or falls) with how fast the *reference* is moving — read from the preview, no
segment identification. `beta=0` reproduces `ff_pi_tuned` bit-for-bit. On 40 worst / 40 typical:

```
beta > 0 (more authority on fast transients):  190 -> 192 -> 215 -> 296 -> 461
beta < 0 (less authority on fast transients):  190 -> 180 -> 201 -> 249 -> 315
```

`beta=0` is a local optimum in **both** directions, and the positive direction — the one §9
predicted — fails hardest. The `beta=-0.1` dip was tested on `ALL[1000:1500]`:

```
beta=+0.00 mean=56.617    beta=-0.10 mean=57.689
paired delta = +1.072 +- 1.595 (SE)  t=+0.67   improved 202/500, worsened 267/500
-> indistinguishable from noise (and the point estimate REVERSES sign vs the subset)
```

**§9's "globally conservative but locally aggressive" claim is wrong.** With ~300 ms of dead time, a
fast-moving reference means the loop is being excited at higher frequency, where the delay's phase
lag is largest — so fast transients are where conservatism is *most* binding, not least. The
detuning is not a compromise averaged over conditions that could be unwound locally; it is
maximally necessary exactly at the moments that generate the tail. `ff_pi_tuned`'s flat authority is
already correct. (Typical segments *do* improve slightly with `beta>0`, 46.86→46.57, consistent with
the same physics: slow targets sit where phase lag is small. The lever exists; it points the wrong
way for the segments that matter.)

### Methodological warning, part 2

Selecting the worst-40 under `beta=0` and then comparing configurations on that set is **biased**:
those segments are extreme *under the baseline*, so regression to the mean makes almost any
perturbation look like an improvement. Both `pole=0.2` (§8) and `beta=-0.1` showed subset gains that
vanished or reversed on untouched segments.

The sweeps remain valid for the large **degradations** they found — observing those despite a bias
toward improvement makes them stronger. But every small apparent gain from a worst-N sweep in this
document is an artifact and was correctly not acted on. Any future candidate must be confirmed by a
paired test on a split not used for selection.

## 11. Conclusion (SUPERSEDED — see §12)

> This section concluded, after eight negatives, that the classical structure was exhausted. **That
> was wrong.** The ninth idea worked, and it is worth being precise about why the conclusion failed:
> all eight negatives were about *tuning* or about *reference/gain shaping* — they treated the plant
> as an unconstrained linear system with a delay. None of them addressed the plant's own
> **constraints**. The search space was blinkered, not exhausted. Kept below as written.

`ff_pi_tuned` sits at a verified local optimum in **every** direction probed: `lam`, `i_clip`,
`lead`, gain-curve shape, gain magnitude, speed-scheduled detuning, dynamic (lead-lag) inverse, and
state-dependent authority. Eight hypotheses, eight negatives.

The classical structure is exhausted, and the reason is now a measurement rather than a guess: the
plant has ~300 ms of **irreducible dead time**. That forces ~2× detuning (delay caps stable loop
gain), caps achievable bandwidth, and is already optimally compensated by `lead ≈ 2–3`. What remains
is a *prediction* problem, and prediction quality over the delay horizon is the whole of `cnn`'s
advantage: 25 steps of learned preview against one tap on a smoothed reference (§9, §1).

Closing the classical thread at **`ff_pi_tuned` = 54.56** on the full 5000. The remaining 7.5 points
to `cnn` are not recoverable by tuning a linear controller; they require the nonlinear preview
policy already on `master`.

Nothing in this branch changes the `cnn` deliverable on `master` (47.87).

## 12. The rate limiter — the constraint never checked, and the first thing that works

Reframe that unlocked this: the plant is **not a physical system**. It is a transformer that
quantizes its own lataccel feedback into 1024 tokens and applies a hard clamp. Its "dynamics" are
whatever the network plus that clamp produce, so classical intuitions can fail for entirely
non-physical reasons. Two things had never been tested: behaviour vs perturbation **amplitude**, and
the plant's own **constraints**.

### The plant is nonlinear, and the rate limiter is why

Impulse response (`plant_impulse.py`) swept over a 50x amplitude range:

```
h[k]/du at k=4:  0.688  0.742  0.431  0.314  0.219  0.171     (du = 0.02 -> 1.00)
implied DC gain:  ~1.79 at du=0.1        ->        ~1.17 at du=1.0
```

Incremental gain **compresses ~35%** with amplitude — the static-gain picture of §8 holds only for
small signals. Cause: at `du=1.0` the k=4->k=5 step is **0.489**, sitting on `MAX_ACC_DELTA = 0.5`.

### The constraint I had wrongly dismissed

§4 "ruled out saturation" by checking `act_sat` — steer reaching +-2. But the binding limit is not the
steer range, it is `MAX_ACC_DELTA`: the plant clamps its own **lataccel change** to 0.5 per step.
That was never measured. It is also the same constraint whose zero gradient once caused the MPC
square-wave bug, so it was known to bite hard here.

The cost makes it devastating. Typical jerk cost ~20 implies RMS `dlataccel` ~0.045/step, so the
clamp sits **11x above normal operation**, and since jerk is charged quadratically at 10000x, one
saturated step costs **25-64x a normal step**:

```
 seg  jerk_cost  sat steps  jerk from sat  share  RMS dlat (unsaturated)
 133      261.9         26          162.9  62.2%                  0.1029
 222      115.6         10           62.7  54.2%                  0.0737
 265      123.3         10           62.5  50.7%                  0.0789
 348      105.0          7           43.8  41.7%                  0.0789
  26      166.2          3           55.4  33.3%                  0.1057
 195       91.9          4           25.1  27.3%                  0.0821
 160       48.9          1            6.3  12.8%                  0.0654
  22      122.3          0            0.0   0.0%                  0.1106
```

**Three to twenty-six individual timesteps produce 27-62% of the jerk cost.**

### This is the whole of cnn's tail advantage

```
 seg | ff_pi cost    jerk  sat |  cnn cost    jerk  sat
 133 |     1994.8   560.2   62 |     232.1   132.8    0
 348 |      466.3   153.8   10 |      87.0    51.4    0
 265 |      367.3   124.1   10 |      82.5    51.2    0
 222 |      249.3   111.1   10 |      71.8    40.9    0
 195 |      213.6    97.2    4 |      96.4    49.4    0
 160 |      184.4    48.8    1 |     150.1    58.6    1
  26 |      189.6    81.6    0 |     160.1    88.1    1
  22 |      284.1    68.4    0 |     282.3    90.3    0
                  total saturated steps:  97  ->  2
```

Where `ff_pi` saturates, `cnn` eliminates it and cost falls **4-8x**. Where saturation is already
zero (segs 22, 26), `cnn` gains **0-1.2x** — and on seg 22, the only segment with zero saturated
steps, `cnn` does not help at all (282.3 vs 284.1). That is the segment §1 already flagged as
genuinely hard. **cnn's tail advantage is saturation avoidance**, and this one mechanism subsumes
§1's concentration, §5's "213 ms lag", the burst structure, and §8's detuning.

### The fix: conditional anti-windup (VERIFIED)

While the clamp binds, the plant cannot respond at any steer, but the PI integrator keeps
accumulating error it cannot act on — so when the clamp releases, the stored command overshoots.
`i_clip` is a **fixed** magnitude clamp and was already optimal (§4); a **conditional** one had never
been tried. Freeze the integrator while the plant is rate-saturated, and hold that freeze for the
duration of the dead time:

```
 hold   40 worst   40 typical            fresh-split paired test (3000 segs, [1000:4000])
    0      189.9        46.86            baseline 53.476
    1      174.2        46.86            52.966   delta -0.510 +- 0.195  t=-2.62
    2      171.6        46.86
    3      155.3        46.86            52.020   delta -1.456 +- 0.372  t=-3.91  (67 up / 40 down)
    5      163.5        46.86
    8      164.6        46.86
```

**`hold=3` is optimal, and 3 steps is exactly the dead time measured independently in §8** — the
freeze must persist as long as the command keeps arriving. Not a fitted coincidence.

Typical segments are **bit-identical** at 46.86 in every configuration: the mechanism fires on only
~3.6% of segments and is inert elsewhere. That is what distinguishes this from the eight negatives,
every one of which traded tail against typical.

### Slew limiting fails — the big moves are a symptom

Capping the steer slew rate to keep the plant off its clamp makes things monotonically worse
(189.9 -> 187.2 -> 214.8 -> 239.5 -> 436.5 as the cap tightens to 0.05). And:

```
              |du| rms    max |du|   lag (steps)
ff_pi_tuned     0.0300       0.296          0.67
cnn             0.0231       0.157          0.33
```

`cnn` moves both **slower and earlier**, and its natural max slew (~0.157) is right where forcing
`ff_pi` to that limit hurt. Large steer moves are a *consequence* of being behind, not the cause —
so limiting them cripples the controller, while removing the windup they cause does not.

### A failed experiment of my own

A first attempt to bound the tail by trajectory optimisation through the differentiable plant
(`tail_floor.py`) ran Adam from **zero** actions over a 400-step horizon, never converged, and
reported "optimal" costs 3x *worse* than the controller it was meant to bound. Those numbers were
discarded. Warm-starting from `ff_pi_tuned`'s own actions works (166.9 by iteration 100 vs 595.9).

### Final numbers (VERIFIED)

Note `data/SYNTHETIC` holds **20,000** segments; the repo's quoted "full 5000" metric is `ALL[:5000]`.
All three bases agree:

```
                                  n       ff_pi_tuned   ff_pi_rl2    delta    bootstrap 95% CI
ALL[:5000]  (repo basis)       5000            54.555      52.297   -2.258   [-3.236, -1.350]
ALL[5000:]  (never evaluated) 15000            54.987      53.101   -1.886   [-2.346, -1.483]
ALL         (20000)           20000            54.879      52.900   -1.979   [-2.396, -1.567]

sign test (20000): 389 improved / 231 worsened of 620 activated, p=2.31e-10 (distribution-free)
median delta among activated: -5.610
consistency: established ff_pi_tuned on ALL[:5000] = 54.56, measured here = 54.555
```

The effect **replicates on 15,000 segments never evaluated by anything in this project**. The claim
does not rest on a t-test: the bootstrap CI excludes zero and the sign test is distribution-free,
which matters because the paired deltas are heavy-tailed (individual segments move by hundreds).

Exactness shortcut used and verified: a segment whose baseline never triggers the clamp cannot
engage the freeze, so its `hold=3` run is bit-identical (checked on 40 non-saturating segments).

Footprint of the mechanism across all 20,000 segments:

```
saturating segments: 626 = 3.13%,  mean 6.8 saturated steps each
cost 300.9 (saturating) vs 46.9 (clean) = 6.4x
the 3.1% saturating segments hold 17.2% of ALL cost
```

**Updated classical ladder on the repo basis:** PID 110.76 -> ff_pi 59.06 -> ff_pi_tuned 54.56 ->
**ff_pi_rl2 52.30**. The gap to `cnn` (47.87) falls from 6.69 to 4.43, a 34% reduction.

### Still open

- CMA-ES re-tune **with** anti-windup returned no improvement (`params={}`), but that is **untested,
  not refuted**: with ~14 activated segments in a 400-segment tune set there is no signal for the
  anti-windup parameters. A real test needs a tune set of several thousand segments.
- Whether `ff_pi_tuned` is a *global* optimum in the original 6 parameters is still untested — the
  multi-start search was killed for being ~10 hours at the achievable throughput.
- The trajectory-optimisation floor reaches **82.3** on the mean plant for the six worst segments
  (vs `cnn` 142, `ff_pi_tuned` 595.9) but 510.5 when its open-loop actions face the sampled plant.
  So large headroom exists in principle; realising it needs closed-loop feedback, not a better
  open-loop sequence.

## 13. Is "hard" intrinsic, or a noise lottery?

Prerequisite for any hard-example mining, and a general caution for this benchmark: segments are
hard *under a particular seed*, since `tinyphysics` draws plant noise from `md5(data_path)`. If
hardness does not survive reseeding, selecting on it selects noise -- the same selection-bias trap
that produced several false positives in §8 and §10.

Test (`hard_stable.py`): symlink identical CSVs under new names -> identical data, different noise
realisation. `cnn` over 800 segments, two seeds:

```
Pearson  r = 0.877      Spearman r = 0.913      log-cost r = 0.936

worst  5% ( 40 segs):  28 shared = 70.0%   (random baseline  5%)
worst 10% ( 80 segs):  53 shared = 66.2%   (random baseline 10%)
worst 20% (160 segs): 111 shared = 69.4%   (random baseline 20%)

individual segment variability: median 14.0%, p90 37.7%
among the 20 worst under seed A: median relative change 24.1%
```

**Hardness is intrinsic.** Spearman 0.913 is the statistic to trust here -- rank-based, so immune to
the heavy tail that inflates Pearson and inflated several earlier claims in this document. Worst-decile
overlap is 6.6x the random baseline.

But roughly **30% of the hard set churns between seeds**, and the worst segments move 24% in cost. So
a frozen hard list is partly noise; any mining scheme must re-score periodically rather than select
once. This also bounds how tightly one should ever fit a "hard set" here.

### Related: cnn is not overfitting noise realisations

Same technique applied to the policy itself (`seed_robust.py`):

```
group                  controller    original  reseeded    delta
trained-on [0:400]     cnn              46.06     45.39    -1.5%
trained-on [0:400]     ff_pi_rl2        49.49     49.78    +0.6%
held-out [5000:5400]   cnn              55.18     54.33    -1.5%
held-out [5000:5400]   ff_pi_rl2        58.77     59.41    +1.1%
```

The diagnostic is the *differential*, not the sign: a policy that had learned specific noise paths
would degrade on reseeded TRAINING segments while held-out stayed flat. That differential is exactly
zero (-1.5% both). `cnn` is robust to noise realisation, so multi-seed training has little to buy.

### Where cnn's remaining cost actually is

```
                          ff_pi_tuned  ff_pi_rl2       cnn
median                          46.45      46.45     44.33
p90                             79.72      79.67     75.91
p99                            297.91     237.69    150.13
max                           1994.78     906.16    761.98
worst 10% share of cost         29.3%      26.8%     23.6%

cnn rate-clamp saturation: 1.57% of segments, 1.3% of its jerk cost
```

Two consequences. First, the anti-windup fix is confirmed as purely tail-targeted -- `ff_pi_rl2`'s
median and p90 are *identical* to `ff_pi_tuned`, with the entire gain in p99 and max. Second, `cnn`
has already solved saturation (1.3% of its jerk cost, so the zero-gradient clamp at `torch_sim.py:129`
is worth only ~0.3 points despite being a real defect), and its remaining cost is **broad and
tracking-dominated** rather than tail-concentrated.

This also corrects §1 as a general claim: `cnn` beats `ff_pi_rl2` at *every* quantile, median included
(44.33 vs 46.45). The "six segments hold 72% of the gap" figure was specific to `ff_pi_tuned` on the
500-segment split; against `ff_pi_rl2` over 3000 segments six segments hold 19.5%, and the median
difference alone is 29% of the gap.

## 14. Hard-example mining: no gain (ninth negative)

Controlled A/B (`mine.py`). Both arms warm-start from `cnn_PM.pt` and share the pool
(`ALL[2000:4000]`), hyperparameters and iteration count; only batch sampling differs — uniform
versus 50% drawn from the current worst decile, re-scored every 50 iterations. Gate passed first
(§13): hardness is intrinsic, so mining targets something real.

Selection set (`ALL[4000:4200]`, torch sim):

```
baseline cnn_PM   mean=48.20  median=45.21
uniform  it175    mean=46.92  median=45.05
mined    it150    mean=47.41  median=45.68     <- worse than baseline's median
```

Real sim, `ALL[5000:6000]` — never trained or selected on by either arm:

```
baseline cnn_PM   mean=49.795  median=44.06  p90=75.19  p99=151.16
uniform  it175    mean=48.969  median=45.46  p90=76.65  p99=135.52
mined    it150    mean=48.917  median=45.34  p90=76.22  p99=125.16

uniform vs baseline: -0.826  95%CI [-2.857,+1.013]  improved 423/1000
mined   vs baseline: -0.878  95%CI [-2.904,+0.874]  improved 320/1000
mined vs uniform (the A/B): -0.052 +- 0.297         improved 317/1000
```

**The A/B is flat**: -0.052 +- 0.297 (t=-0.18). The sampling strategy contributed nothing.

**Neither fine-tune is an improvement either.** Both mean deltas have 95% CIs spanning zero, and the
per-segment counts are decisive against them: uniform improves 423/1000, mined only 320/1000 —
eleven standard deviations from a fair coin in the WRONG direction. Both make most segments slightly
worse and recover it from a few large tail wins. The quantiles show the trade directly: median and
p90 worsen (44.06 -> 45.46/45.34, 75.19 -> 76.65/76.22) while p99 improves (151 -> 136/125).

Mining does work as designed — it produced the best p99 (125.16) and improved the pool's hard-set
mean 114.2 -> 102.8 — it simply costs more in the bulk than it returns. That is the arithmetic
predicted before the run: **76% of cnn's cost sits outside its worst decile**, so a small regression
there cancels a large tail win. Mining is the right tool for a tail-concentrated problem, which is
`ff_pi`'s shape, not `cnn`'s.

Also retracted: the uniform arm's 1.28-point gain on the selection set led me to suggest `cnn_PM.pt`
was under-trained. It does not replicate as a clean win on fresh segments (CI spans zero, 423/1000).

### Operational note

The first real-sim evaluation ran without `OMP_NUM_THREADS=1`: 12 workers x 16 torch threads on 32
cores, load average 176, ~2.5 h per 1000 segments. With threading pinned and 24 workers the same
three evaluations took **about 3 minutes** — a ~50x speedup — and reproduced the slow run's numbers
exactly (baseline 49.795 vs 49.794). Every CPU-sim script here should pin thread counts.

## 15. Where the floor actually is

`cnn` sits at 47.87 and the honest leaderboard frontier is ~36. Neither number says how much room
exists. Two attempts to establish that:

### Attempt 1: perfect-model MPC — FAILED, and instructively

`floor_mpc.py` replans every step by gradient descent through the *actual* TinyPhysics network, then
executes against the sampled plant. First run reported a "floor" of **2116** with 31% rate-clamp
saturation — started Adam from zero actions, exactly the mistake already recorded for `tail_floor`
in §12 and repeated anyway. Warm-starting from the inverse-plant feedforward fixed convergence
(planned cost drops 86.8 -> 48.4 per step) and gave **432**.

Still useless as a floor, and the reason matters: **planned ~50, realised 432**. Optimising against
the expected-mode plant exploits a deterministic fiction the sampled plant does not follow. This
repo has now hit that trap three times — the README records the expected-value plant being biased
(PID 29 expected vs 68 sampled), and the training work found "expected-mode gradient overfit the
surrogate (train 23, sampled 55) -> use gumbel mode". Any planner here must be optimised against
*sampled* rollouts. Fixing this properly needs scenario MPC over K realisations.

### Attempt 2: measure the noise directly — WORKS

`noise_floor.py`. Drive the plant with an *identical* fixed action sequence under two seeds; the
difference is pure noise, with no control feedback involved:

```
per-step noise std        = 0.2061
noise INCREMENT std       = 0.0337     (matches the ~0.030 conditional std measured earlier)
increment lag-1 autocorr  = +0.042     (white increments -> random walk confirmed)

IRREDUCIBLE JERK COST     = 11.33

  lat floor at feedback lag 1 step :  5.67   -> total floor 17.00
  lat floor at feedback lag 2 steps: 11.80   -> total floor 23.13
  lat floor at feedback lag 3 steps: 17.62   -> total floor 28.96   <- measured dead time
  lat floor at feedback lag 4 steps: 22.95   -> total floor 34.29
```

**The floor is ~29** at the 3-step dead time measured independently in §8. Noise enters lataccel
*after* the action, so its jerk contribution cannot be removed by any controller; and feedback
cannot correct the random walk faster than the dead time, which sets the tracking floor.

**Cross-validation.** The honest frontier (~36) sits 7 points above this independently derived floor.
Two unrelated facts — a noise measurement here, and what the best honest entry achieved — landing
that close is mutual corroboration, and implies the frontier entries are near-optimal rather than
exploiting something.

### Consequences

- `cnn`'s headroom is **~19 points**, of which ~12 is demonstrably reachable (frontier 36).
- **56% of cnn's jerk cost is irreducible** (11.33 of 20.4); only ~9 points of its jerk is
  controllable at all.
- The remaining opportunity is almost entirely **tracking**, and it is bounded by effective lag.
  Reducing effective feedback lag from 3 steps to 2 is worth ~6 points on its own (28.96 -> 23.13).
  That is what better *prediction* buys — i.e. what MPC on an accurate model does, and why the
  frontier was reached that way.
- Tuning is closed: nine negatives, and `ff_pi_rl2` is at a verified local optimum in every probed
  direction. The remaining work is model-based.

## 16. Scenario MPC through the real network: the solver is the bottleneck

Given §15 (linear surrogates are capped at ~1.35x the noise floor), the right move is to stop
fitting models and plan through the **exact** model — the network itself. `scenario_mpc.py` does
that: at each step it optimises one action sequence against **K independent noise draws
simultaneously**, using straight-through Gumbel sampling so the stochastic path stays
differentiable, plus a straight-through rate clamp so gradients survive saturation.

**The scenario averaging works.** Planning in expected mode gave planned ~50 / realised 432 —
optimising against a deterministic fiction. With K=4 sampled scenarios the planned cost (370–700)
now *exceeds* the realised cost (287–428), i.e. the plan is honest about noise rather than exploiting
its absence. That trap is solved.

**But the controller is still bad, and the reason is the solver, not the model.** Measured against
its own initialisation:

```
FF warm start alone (scale 1.00)   461.88
FF warm start alone (scale 1.79)   577.42
scenario MPC (H=12, K=4, 6 iters)  ~426
```

Six Adam iterations at lr=0.05 on a 12-dimensional action vector, with gradients propagated through
12 stochastic Gumbel steps, improves on its own warm start by ~8%. The gradient noise swamps the
descent. The model is exact and the objective is right; the inner optimisation is simply too weak.

Two side results worth keeping:

* **The detuning does not transfer to a pure feedforward.** `ff_pi_tuned`'s 1.79x detuning is optimal
  *because its PI integrator supplies the withheld authority*. With no integrator, raw inversion is
  better (461.88 vs 577.42). I patched the warm start to "fix" the missing 1.79 and measured it as a
  regression before shipping — worth stating because the detuning result (§8) reads as a property of
  the plant when it is really a property of the plant *plus* an integrator.
* **This was never going to be a submittable controller** — seconds per step, and it calls the plant
  as an oracle. It was only ever a way to bound the floor, and §15 answered that more cheaply.

### Status of the MPC route

Not refuted — **unfinished**. The obstacles are now separated: model fidelity is solved (use the
network), expected-mode exploitation is solved (scenario averaging), and what remains is a
well-engineered inner solver — more iterations, longer horizon, gradient variance reduction, or a
structured/analytic step instead of generic Adam. That is a scoped project, not a bolt-on.

## 17. Capacity is not the limit either (tenth negative)

The released controller is `AblNet('PM')` with **11,243 parameters**, against a plant that is a ~1M
parameter transformer, and §15 showed the shortfall is broad (median 44.33 vs a ~29 floor) rather
than tail-shaped. A broad shortfall from an 11k model is the textbook signature of a capacity limit,
and every previous attempt had been a fine-tune *around* that small model, so the variable had never
been changed.

From-scratch A/B (`cap_ab.py`), identical data / curriculum / batch / schedule / 400 iterations,
differing only in width. The small arm was retrained rather than reusing `cnn_PM.pt`, so the
comparison could not confound capacity with a different training recipe.

```
                            torch val (200 segs)      real sim ALL[5000:6000], 1000 segs
                            best mean   median        mean     median   p90     p99
released cnn_PM   (11k)          --        --        49.795    44.06   75.19   151.16
retrained small   (11k)       48.62     45.98        49.750    45.93   79.70   144.72
retrained big     (89k)       48.30     46.29        50.075    45.60   79.12   140.17

big - small: +0.325   95%CI [-0.792, +1.854]   t=+0.48
```

**Eight times the width converges to the same place**, and is marginally worse in the real sim.

The early signal was misleading in an instructive way: at iteration 50 the wide model was ~2x better
on both mean and median (53.69/49.05 vs 101.65/72.57). That was **learning speed, not final
quality** -- it reached the same ceiling sooner and stopped. Anyone reading a capacity experiment off
an early training curve here would conclude the opposite of the truth.

Validation that the experiment was sound: the retrained small arm (49.750) reproduces the released
checkpoint (49.795) on mean, so the recipe is faithful and 400 iterations is converged.

**What this rules out.** More capacity is a strictly larger policy class, so this closes capacity
*and* policy-class expressiveness together. The limit lies in what the policy can observe, or in the
training signal -- not in the model.

### Running tally of cnn improvement attempts, all null

| attempt | result |
|---|---|
| hard-example mining (§14) | A/B flat, -0.052 +- 0.297 |
| uniform fine-tune, bigger pool (§14) | CI spans zero, 423/1000 improved |
| multi-seed training (§13) | ruled out -- no noise memorisation |
| saturation-aware training (§13) | ~0.3 pts; cnn already avoids saturation |
| **network capacity, 8x width (§17)** | **+0.325, CI spans zero** |

Still untested, with weaker priors than capacity had: gradient-variance reduction (effective batch
is 32 segments with gradients through a *stochastic* plant -- and §16 showed exactly that noise
destroying the MPC inner solver), and preview length (H=25 of 50 available, though with only ~3
steps of dead time the marginal value is doubtful).
