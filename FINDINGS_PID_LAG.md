# PID with dead-time compensation (Smith predictor): worse, for a textbook reason

## Setup

The plant's measured impulse response, normalised to unit DC gain:

```
lag k     0     1     2     3     4     5
h[k]    0.00  0.02  0.08  0.22  0.38  0.30      onset ~2 steps, bulk and peak at 5
```

Dead time is what caps loop gain on this plant -- it is why `ff_pi_tuned` commands roughly half the
steer that true plant inversion calls for. The classical fix is a **Smith predictor**: run the plant
model with and without its delay and feed the controller

    feedback = y_measured + (y_model_undelayed - y_model_delayed)

so the loop effectively sees a delay-free plant and can be tuned as if there were no lag.
`controllers/pid_lag.py` implements this using the measured impulse response as model taps, scaled
by the verified `G(v)`. `smith=0` disables prediction and reproduces comma's stock PID bit-for-bit.

Gains were tuned SEPARATELY per arm (CMA-ES, 400 segments, 30 iterations, identical search space).
That matters: a Smith predictor changes the loop's effective dynamics, so comparing at fixed gains
measures nothing -- the whole point is that it should permit gains a delayed loop cannot afford. At
stock gains `smith=1` scored 122 vs 103, which is expected and uninformative.

## Result

```
held-out ALL[500:620]        defaults    tuned
  SMITH=0  plain PID          112.19     121.62      (+9.43 -- tuning OVERFIT)
  SMITH=1  Smith predictor    145.25     128.62      (-16.63)

tune-set (400 segs) best:      87.01      87.74      -- nearly identical
```

Best plain PID on held-out is **112.19**; best Smith-predictor PID is **128.62**. Dead-time
compensation costs ~16 points. Note the tune-set scores are indistinguishable (87.01 vs 87.74) while
held-out differs by 7 -- another instance of this metric's subset sensitivity.

## Why: the right fix for the wrong problem

A Smith predictor improves **setpoint tracking** through dead time, and is well known to **degrade
disturbance rejection**: the disturbance still traverses the delay, and the predictor's inner loop
opposes the integral action that would otherwise reject it.

This plant is disturbance-dominated. The noise is a random walk with lag-1 autocorrelation 0.98, and
of the ~29 achievable floor about 11.3 is irreducible noise. The project's earliest diagnosis said
the same thing from the other direction -- because the disturbance is slow drift rather than
high-frequency jitter, **integral feedback is the key lever** and low-pass filtering is useless.

So this applied a setpoint-tracking remedy to a disturbance-rejection problem. The dead time is real
and does cap loop gain, but Smith prediction is not the way to recover it here.

## Secondary finding: PID tuning overfits where ff_pi tuning does not

The plain-PID arm tuned to 87.01 on 400 segments and came out **worse than untuned defaults** on
held-out (121.62 vs 112.19). `ff_pi` tuned cleanly on the identical budget and produced a real
held-out gain. A feedback-only controller's mean cost is dominated by the segments it handles badly,
which makes its tuning landscape far more subset-sensitive.

---

# Lookahead error: the same problem, solved the other way round

Instead of predicting the plant's output (Smith), shift the *reference*: measure the error against
the target a few steps AHEAD, leaving the feedback signal as measured lataccel.

```python
e = target_future[k] - current_lataccel          # k ~ 2, fractionally interpolated
```

## Result: 24% off comma's baseline PID, from one line

```
ALL[:5000]                 mean     median    p90
  stock PID              110.756    73.67   173.52
  PID + 2-step lookahead  84.117    61.76   118.03

  delta -26.639   95% CI [-29.190, -24.050]   improved 4537/5000  (90.7%, ~57 SD)
```

Held-out sweep at *unchanged* gains, showing a clean unimodal optimum:

```
  k0     0.0    1.0    2.0    3.0    4.0    5.0    7.0
  cost 112.19  91.87  82.99  85.24 101.80 109.05 181.17
```

## Why this works where the Smith predictor failed

Both compensate dead time; they differ in *what they modify*.

* **Smith predictor** substitutes a model prediction into the FEEDBACK signal. It is well known to
  degrade disturbance rejection, and this plant is disturbance-dominated (random walk, lag-1
  autocorrelation 0.98; ~11.3 of the ~29 floor is irreducible noise). Cost: **-16 points**.
* **Lookahead** changes only the REFERENCE. Feedback remains measured lataccel, so integral action
  rejects drift exactly as before, and the loop merely aims where the target will be when the action
  lands. Gain: **+26.6 points**.

On a disturbance-dominated plant that distinction decides the outcome. The textbook remedy lost to
the naive one because the textbook remedy targets setpoint tracking.

## Lookahead SUBSTITUTES for feedforward — it does not complement it

Applying the identical change to the feedback path of `ff_pi_rl2` (best classical, 52.30) is
monotonically harmful:

```
  fb_look   0.0    1.0    2.0    3.0    4.0    6.0
  held-out 53.41  56.08  61.29  68.58  79.75 107.52
```

`ff_pi` already anticipates: its feedforward inverts the plant against `c[k0+lead]`. Adding
lookahead to the feedback double-counts the anticipation, the controller turns too early, and the
two channels fight. PID has no feedforward, so the error term is its only route to anticipation --
which is exactly why it gains so much.

**Three independent results agree the right anticipation is ~2 steps**: `ff_pi`'s `lead` tunes to
2-3, the PID lookahead optimum is 2, and joint tuning drove `k0`->0 with `kv`~2.25 (~2 at typical
speed). All well short of the plant's 5-step bulk delay -- anticipating further means committing to
a target that has not arrived.

## The speed/acceleration schedule does not earn its parameters

`k = k0 + kv*(v/30) + ka*a`, jointly tuned with the gains, scored **83.65** on held-out against
**82.99** for a plain constant -- and set `k0` to ~0 with `kv` ~ 2.25, rediscovering "about 2" the
long way round. Consistent with the impulse-response measurement: bulk response timing is
speed-INVARIANT at ~5 steps and only the onset moves (4 steps at low speed -> 2 at high, r = -0.98).
The constant is the whole effect.

## Scope

This improves the *baseline*, not the deliverable. Every controller here that already has
feedforward gets nothing, or is harmed. It is worth recording because the reference PID everyone
benchmarks against leaves ~26 points on the table for a one-line change, and because the
Smith-vs-lookahead contrast cleanly identifies what kind of compensation this plant admits.

---

# Velocity-scheduled lookahead, derived from the measured response time

## The plant's response time, measured

Step response to a +1.0 m/s^2 request (steer step `1.0/G(v)` per segment, on-manifold baseline,
expected mode, 384 segments, dt = 0.1 s):

```
 v band (m/s)   G(v)   t10%    t50%    t90%    clamp?
   0.0 - 18.1  1.431  400ms   500ms   700ms     no
  18.1 - 26.8  1.485  400ms   500ms   600ms     no
  26.8 - 30.9  1.580  300ms   400ms   500ms     no
  30.9 - 37.2  1.670  300ms   400ms   500ms     no
```

The rate clamp does NOT bind for a 1.0 step (peak 0.431 vs 0.5). This also **corrects** an earlier
claim: response timing is NOT speed-invariant. That claim came from the impulse-response *peak*
location; the *step* response — which is what "time to answer a target change" means — falls from 7
steps to 5 across the speed range.

## Result: -28% off the stock PID with UNCHANGED gains

`pid_phys.py` sets the lookahead from those measurements, `k(v) = scale * (a + b*v)`, and leaves
`p=0.195, i=0.100, d=-0.053` exactly as shipped (verified identical, no integral clamp).

```
ALL[:5000]                                  ALL[5000:6000]  (clean, no selection)
  stock PID            110.756                  114.645   median 72.80  p90 192.86
  constant k=2          84.117                   85.155   median 61.37  p90 130.47
  v-sched t90 x0.4      79.946                   81.132   median 60.71  p90 120.45

  v-sched vs stock    :  -33.513  95%CI [-41.05,-27.68]  improved 910/1000
  v-sched vs constant :   -4.023  95%CI [ -6.03, -2.11]  improved 697/1000
```

Hyperparameters (`basis`, `scale`) were selected on `ALL[500:620]`, which is *inside* `ALL[:5000]` —
so the headline split is mildly contaminated. `ALL[5000:6000]` is fully disjoint and the effect is
unchanged (-4.023 vs -4.171), so the selection bias was negligible.

## The optimum is ~40% of the response time, not 100%

```
 scale    t10      t50      t90
  0.40   87.35    84.64   *81.40*
  0.50   85.01   *83.08*   84.34
  0.60  *82.83*   83.52    93.48
  1.00   89.22   100.67   148.37     <- full response time: worse than stock PID
```

Three independent definitions of response time, each at its own optimum, converge on **2.1-2.4
steps** — the same value a free tuner found and the same as `ff_pi`'s independently-tuned `lead`.
Anticipating the *full* response time is catastrophic.

## Why: closed-loop lag, not open-loop settling time

The open-loop step response assumes a single held action. In closed loop the controller re-acts every
100 ms, so successive corrections do most of the work. Measuring the shift that maximises
`corr(achieved, target)`:

```
controller              closed-loop lag   RMS err
  stock PID                  3.65 steps    0.1180
  PID + lookahead k=2        1.74          0.0986
  PID + v-sched              1.47          0.0967
  ff_pi_rl2                  0.81          0.0769
  cnn                        0.14          0.0725
```

* closed-loop lag is **3.65 steps**, not the 5-7 step open-loop settling time;
* lookahead buys lag reduction nearly 1:1 (2 steps of lookahead: 3.65 -> 1.74);
* **RMS error is monotone in lag across every controller in this project** — the whole hierarchy is
  explained by how far behind the target each one runs;
* `cnn` tracks essentially in phase (0.14), which is *why* it wins.

The optimum sits at lag ~1.5 rather than 0 because aiming further ahead reduces phase lag but commits
to a target that has not arrived. `cnn` reaches 0.14 without paying that penalty because it consumes
the whole 25-step preview instead of a single tap — the advantage a single-tap lookahead cannot copy.

## Correction to an earlier conclusion of mine

I previously concluded a velocity schedule "does not earn its parameters", based on jointly tuning
`k = k0 + kv*(v/30) + ka*a`, which returned `k0~0, kv~+2.25` — lookahead **increasing** with speed —
and no gain. That has the **wrong sign**: the plant responds *faster* at high speed, so less
anticipation is needed there. The physics-derived schedule decreases with speed (2.92 -> 1.96 steps)
and is worth -4.0 points. CMA-ES searched into the wrong basin, almost certainly the under-searching
documented above (`sigma=0.6` returning nothing where 0.15 finds gains).

Deriving the schedule from measurement rather than searching for it is what made the difference.

## Scope

This improves the **baseline**, not the deliverable. `ff_pi_rl2` (52.30) and `cnn` (47.87) already
anticipate via feedforward, and adding feedback lookahead to `ff_pi_rl2` is useless — a fine search
drove `fb_look` to 1e-06. 79.95 is a much better PID, not a competitive controller.

---

# Target smoothing, and four things that did not stack on top

## Tikhonov smoothing of the target: verified, -3.0

`pid_smooth.py`. Not a Kalman filter -- the target is exact, so there is no state to estimate from
noisy measurements. What is wanted is the tracking-vs-jerk TRADE-OFF, which has a closed form for
this cost: `(I + lam*D'D) c = tau` with `lam = W_jerk/W_track = 2`. Solved causally over
[past | now | preview] by the Thomas algorithm; the lookahead then indexes the smoothed reference.

```
CLEAN ALL[5000:6000]      total   lataccel   jerk    median
  lookahead only         81.132    55.57    25.56    60.71
  + Tikhonov lam=2       78.148    54.85    23.29    60.69
  delta -2.984  95%CI [-5.117,-1.252]  improved 498/1000
```

The sweep optimum lands exactly at the analytic `lam=2`, not at a tuned value -- theory confirmed.
Both cost terms improve. **Caveat: 498/1000 improved is a coin flip and the median does not move**, so
the mean gain comes from taming extremes rather than improving typical segments. The benchmark is the
mean, so it counts, but the two statistics disagree and the sign test is the robust one.

## Reference governor: REJECTED (looked like -4, was +1.7)

Rate-limit the reference so it never demands more than the plant can deliver. On the tuning split it
looked excellent (80.72 -> 76.76, both terms better). On clean data:

```
  + ref governor  79.806  vs  78.148 without    +1.658  CI [-0.740,+4.189]  improved 39/1000
```

**39 of 1000 segments improved.** Pure overfitting, and the warning was visible in the sweep before
verification: `max_rate` 0.15 -> 97.70, 0.18 -> 77.60, 0.20 -> 76.76, 0.22 -> 78.70. A 21-point swing
across a 0.03 parameter change is a knife-edge optimum, and `max_rate=0.20` was selected on exactly
the split the gain was reported on.

Mechanistically it is redundant with Tikhonov: both limit reference movement (one by curvature
penalty, one by hard slew cap), so stacking them over-constrains -- tracking rises 54.85 -> 56.76
while jerk barely improves.

## Latch (hold reference until reached): REJECTED, catastrophically

```
  tol=0.02  2324.27      tol=0.05  1686.38      tol=0.10  647.12      tol=0.20  135.52
```

Lataccel cost explodes to 2244 because holding means aiming at a stale target while the world moves
on, and the staircase reference raises jerk too (23.63 -> 80.04). Same argument that killed
intermittent control here: quadratic jerk plus a never-stationary target makes discrete holding
strictly worse than continuous.

## Pending-response correction: REJECTED, and my prediction was inverted

Discount lataccel still owed by in-flight action increments, `pending = G(v)*sum (1-cum[m])*du[t-m]`.
Applied to the integral path (`pred_i`) or the proportional path (`pred_p`):

```
  pred_i  0.00 -> 80.72    0.25 -> 82.95    0.50 -> 86.13    1.00 -> 97.42
  pred_p  0.25 -> 80.86    0.50 -> 80.88    1.00 -> 82.86
```

I predicted the integral path would HELP (removing windup on already-corrected error) and the
proportional path would hurt. The opposite: the integral path is **eight times more damaging**.

The mechanism works as designed -- jerk falls monotonically 23.63 -> 20.26, so the compounding is
real and discounting it does remove jerk. But tracking explodes 57.09 -> 77.16, because on this plant
the integrator's job is *drift rejection*: the disturbance is a random walk with lag-1 autocorrelation
0.98. **Integral action is the load-bearing element, not the compounding culprit.** This closes a loop
with two earlier results -- the Smith predictor lost 16 points for the same reason, and `i_clip` was
already optimal. Anything that reduces effective integral gain trades away more tracking than the
jerk it buys.

## Velocity-scheduled FEEDFORWARD lead on ff_pi_rl2: null

The idea that gave the PID -33 points, applied to `ff_pi_rl2`'s feedforward tap (constant `lead=2`
-> scheduled 2.92 steps at 5 m/s to 1.90 at 37 m/s). Optimum again at t90 x 0.4 -- independent
corroboration that the plant wants ~40% of its settling time. But:

```
  CLEAN ALL[5000:6000]  54.571 -> 54.419   -0.152  CI [-1.006,+0.738]  improved  412/1000
  HEADLINE ALL[:5000]   52.301 -> 52.181   -0.120  CI [-0.507,+0.274]  improved 2101/5000
```

Both CIs span zero and FEWER than half the segments improve. `ff_pi`'s hand-tuned constant was
already right.

## The principle these four share

**Anticipation is a single resource.** Once a controller holds roughly the right amount, adding more
through a different mechanism gains nothing:

| attempt | result |
|---|---|
| feedback lookahead on `ff_pi` (`fb_look`) | null -- fine search drove it to 1e-06 |
| reference governor on top of Tikhonov | harmful -- two reference limiters |
| velocity-scheduled feedforward lead on `ff_pi` | null -- constant already correct |

This also explains the *size* of the PID win: it gained 33 points because it had **no** anticipation,
not because velocity scheduling is powerful in itself.

## Final state of this branch

```
ALL[5000:6000] (clean)                        total
  stock PID                                  114.645
  + velocity-scheduled lookahead              81.132   -33.5   verified
  + Tikhonov smoothing lam=2                  78.148   - 3.0   verified
  + reference governor                        79.806   + 1.7   rejected
  + pending-response correction               82.95+   + 2.2   rejected
  gain tuning on top of the above             86.96    + 5.6   rejected, overfits

  ff_pi_rl2 (unchanged, still the best classical)   54.571 clean / 52.301 headline
```

**78.148 is the result: a 32% improvement on the shipped PID baseline** from two physically-derived
mechanisms with gains untouched -- and gain tuning on top makes it worse. None of it transfers to
`ff_pi_rl2` or `cnn`, both of which already carry feedforward.

---

# Bootstrapped integrator: -38% on the stock PID, and the biggest win of this branch

## The problem

A feedback-only PID cannot know how much steering a new target needs, so the required steady-state
offset has to be DISCOVERED by accumulating error. A trace through a real corner shows the cost --
error sits at +0.13 to +0.41 for seven consecutive steps while steering creeps 1.129 -> 1.146.

## The fix

The plant model already says what steering a reference needs: `u_model = (ref - roll)/G(v)`. Nudge
the integrator toward the value that would produce it:

```python
integ += boot * (u_model / ki - integ)
```

## Result: verified on both splits

```
ALL[:5000]                total   lataccel   jerk   median     p90
  stock PID              110.756    85.25   25.51   73.67   173.52
  + lookahead + smooth    77.892    54.72   23.17   60.71   110.19
  + integ bootstrap       68.412    46.91   21.50   55.49    88.89

  bootstrap vs smooth  -9.480  95%CI [-11.676, -7.475]  improved 3405/5000
  full stack vs stock -42.344  95%CI [-45.748,-39.153]  improved 4598/5000  (92%)

CLEAN ALL[5000:6000] (never used for selection)
  + lookahead + smooth    78.15
  + integ bootstrap       69.46   -8.692  95%CI [-11.570,-5.766]  improved 680/1000
```

Every quantile improves together, both cost terms improve, and gains are exactly stock.

**Answering the jerk question directly: bootstrapping costs no jerk.** Jerk *improves*, 23.29 ->
21.61. The plant spreads a steering change over ~5 steps behind its rate clamp, so a gentle
model-based nudge is free on that axis.

## It beats conventional feedforward, and the reason is structural

```
  ffw=0.25 (true gain)  82.13     ffw=0.25 (detuned 1.79)  75.27     boot=0.02  65.68
  ffw=1.00 (true gain) 169.88     ffw=1.00 (detuned 1.79) 103.33
```

Parallel feedforward adds to the output unconditionally, so when the gain model is wrong the error
persists and the integrator must fight it. The bootstrap nudges the integrator TOWARD the model, so
it is a **soft, self-correcting prior**: the model supplies most of the offset immediately and
feedback remains free to overrule it. On a plant whose gain model carries +-9% error that is the
better structure.

Confirmed by the calibration each one wants: **open-loop feedforward needs DETUNING (1.79) while the
bootstrap wants the TRUE gain (1.0-1.4, with 1.79 worse).** Opposite requirements, exactly as the
self-correcting reading predicts.

## What it actually is -- a correction to the original framing

`boot=0.02` is a blend rate, so its time constant is 50 steps = 5 s. That is not a jump-start on
target changes. Gating discriminates:

```
  gate=always     65.68
  gate=steady     68.30    retains most of the value
  gate=transient  73.65    loses most of it
```

The value lives mostly in the **steady-state** contribution. So this is primarily a standing model
prior anchoring the integrator, with a smaller transient benefit on top -- not the "bootstrap after a
target change" it was conceived as. Both contribute; steady dominates.

## Nothing is meaningfully tunable

* `boot` sits in a flat basin -- 0.01/0.015/0.02/0.025/0.03 give 67.6/66.3/65.7/67.4/68.9. The
  apparent knife edge in the coarse sweep (0.05 -> 85.21) was simply past the basin.
* `gate` should be off; `always` wins.
* `gain_scale` should be the measured physical gain; 1.4 beats 1.0 by 0.24, inside noise.

Every value is principled rather than fitted, and gain tuning on top of the stack **overfits**
(86.96 held-out vs 81.40). That is a robustness property, not a missed opportunity.

## Why this one worked when the pending-response correction failed

Both use the same plant model and both touch the integral path, with opposite signs. The
pending-response experiment *reduced* effective integral action and cost 16 points; this *accelerates*
it toward a model estimate and gains 9. Consistent with the principle established three times over on
this plant: the disturbance is a random walk (lag-1 autocorrelation 0.98), integral action is the
load-bearing element -- **help it, do not discount it.**

## Final state of this branch

```
ALL[:5000]                                     total
  stock PID                                   110.756
  + velocity-scheduled lookahead               79.946    (84.117 is the CONSTANT k=2 arm, not this)
  + Tikhonov smoothing lam=2                   77.892
  + bootstrapped integrator                    68.412    -38%, gains untouched

  ff_pi_rl2 (still the best classical)         52.301
  cnn (deliverable, untouched)                 47.872
```

---

# The bootstrap transfers to ff_pi_rl2: 52.301 -> 51.222

Three anticipation mechanisms came back null on `ff_pi_rl2` (feedback lookahead, reference governor,
velocity-scheduled feedforward lead) because it already anticipates. The bootstrap is different: it
changes how the INTEGRATOR is anchored rather than adding anticipation.

And there is a concrete reason it has room. `ff_pi_rl2`'s feedforward is deliberately detuned 1.79x,
so per unit of `(ref - roll)` it emits `0.384` where the true gain needs `0.688` -- leaving **~44% of
the required steering for the integrator to discover by accumulating error**. Same slow-discovery
problem as the PID, smaller in magnitude (44% rather than 100%).

So the bootstrap target is the residual, not the whole command:

    integ_target = ((ref-roll)/G_true(v) - ff) / ki

Consistent with the smaller residual, the optimal blend rate scales down the same way: **0.005 here
vs 0.02 on the PID**.

```
CLEAN ALL[5000:6000]        total   lataccel   jerk   median
  ff_pi_rl2 (boot=0)       54.571     34.01   20.56    46.38
  + boot=0.005             52.930     32.69   20.24    45.23
  + boot=0.010             53.092     32.79   20.31    45.06
  boot=0.005 vs 0: -1.641  95%CI [-2.548,-0.876]  improved  681/1000
  boot=0.010 vs 0: -1.478  95%CI [-2.459,-0.445]  improved  639/1000

HEADLINE ALL[:5000]
  ff_pi_rl2 (boot=0)       52.301     32.12   20.18    46.78
  + boot=0.005             51.222     31.30   19.92    45.66
  + boot=0.010             51.086     31.19   19.89    45.54
  boot=0.005 vs 0: -1.080  95%CI [-1.506,-0.634]  improved 3395/5000
  boot=0.010 vs 0: -1.215  95%CI [-1.716,-0.669]  improved 3135/5000
```

Both CIs exclude zero on both splits, 68% of segments improve on both, and tracking, jerk and median
all move together. `boot=0.005` is chosen over `0.010` because it wins on the CLEAN split and has the
better sign test (681 vs 639); the two are inside each other's CIs, so 0.005-0.010 is a basin.

## Updated classical ladder

```
ALL[:5000]                                  total
  stock PID                                110.756
  PID + lookahead + smooth + bootstrap      68.412
  pid_w_ff (ported reference)                59.49
  ff_pi                                      59.06
  ff_pi_tuned                                54.56
  ff_pi_rl2                                  52.301
  ff_pi_boot                                 51.222   <- best classical
  cnn (deliverable, untouched)               47.872
```

## Open follow-up

`ff_pi_rl2`'s `gain_scale` (1.79) and `ki` were tuned WITH the integrator discovering that residual by
accumulation, so they are co-adapted to its absence. Retuning them with the bootstrap active could
compound -- possibly toward less detuning, since the integrator now reaches its target faster.
Caveat: gain tuning on this controller family has overfit every time it has been tried (plain PID
121.62 vs 112.19 untuned; pid_phys 86.96 vs 81.40), so it needs the fine `sigma=0.15` search and the
held-out guard, and the held-out number is the only one worth believing.

---

# Headline ALL[:5000] sweep of every arm, and two false alarms it raised

Every experiment on this branch re-measured on the same split, so the negatives are directly
comparable to the wins rather than living on assorted held-out subsets. Full table in the README.

Two rows initially looked like they REVERSED an earlier rejection. Both were reading errors, and
both are worth recording because the failure modes differ.

## False alarm 1 -- pid_pend "improves" by 1.75 on the mean

    pid_pend pred=0 (== pid_smooth)   77.892
    pid_pend pred_p=0.25              76.143
    pid_pend pred_i=0.25              78.195

    pred_p=0.25 vs 0   mean -1.748 [-3.22,-0.38]  median +0.3763  better 1813/5000
    pred_i=0.25 vs 0   mean +0.303 [-1.49,+1.85]  median +2.7666  better  614/5000

`pred_p` shows a -1.748 mean with a CI excluding zero -- and is clearly HARMFUL: the median is
+0.3763 and only 1813/5000 (36%) of segments improve, so ~64% get worse while a few large wins drag
the mean down. `pred_i` is unambiguous at 614/5000 (12%). The original rejection stands, and the
mean alone would have reversed it. Third instance this session of a mean-vs-median split (see also
the dualcnn checkpoint screen and pid_wff_v segment 00522).

## False alarm 2 -- "a constant lookahead beats the velocity schedule"

Not a measurement error, a transcription error: `84.117` is the **constant k=2** arm, and the
velocity schedule is **79.946**. Re-measured to settle it:

    pid_phys (t90 schedule, scale 0.4)  79.946
    pid_look (constant k0=2.2)          81.765
    constant vs schedule   mean +1.819 [+0.35,+3.49]  median +0.1627  better 2132/5000

The schedule wins by 1.82, consistent in sign with the -4.023 recorded on the 1000-segment split.
The "Final state of this branch" summary block above had propagated the same transcription error and
is now corrected in place.

Correct PID ladder on ALL[:5000]: 110.756 -> 79.946 -> 77.892 -> 68.412.
