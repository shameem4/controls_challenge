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
