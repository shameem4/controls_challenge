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
