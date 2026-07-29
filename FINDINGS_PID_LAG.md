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
