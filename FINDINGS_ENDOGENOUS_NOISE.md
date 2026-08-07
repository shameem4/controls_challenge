# The noise floor is endogenous — but we are already sitting at it

Prompted by reframing the control action as a *continuation of the token/action history* rather than a
scalar knob. The plant is an autoregressive transformer conditioned on 20 steps of (action, roll, v, a)
and 20 quantised lataccel tokens, so an action is not a lever on an output — it is an entry in the
context that conditions the entire next distribution, including its spread.

That matters because every floor in this project assumed the opposite. The causal floor
`sigma^2 * 26614` (31.24 at the population variance, per-segment in `segfloor.py`) treats `sigma^2` as
exogenous: the controller moves the conditional MEAN, never the conditional SPREAD.

## The assumption is false

Zero-mean jitter added to a controller's actions, in genuine closed-loop rollouts so the context is
always self-consistent (128 pristine segments):

| arm | E[Var] | implied floor | mean abs du | track | jerk | total |
|---|---|---|---|---|---|---|
| `cnn_v4` raw | 0.001075 | 28.60 | 0.01281 | 28.63 | 21.67 | **50.30** |
| + jitter 0.02 | 0.001398 | 37.21 | 0.02019 | 52.25 | 28.88 | 81.13 |
| + jitter 0.05 | 0.002879 | 76.62 | 0.03873 | 81.79 | 59.84 | 141.63 |

A rougher controller gets a genuinely noisier plant: 0.02 of action jitter raises the conditional
variance 30% and lifts its own floor from 28.6 to 37.2. **The floor is a property of the
controller-plant pair, not of the plant alone.**

This is not the incoherent-context artefact it could have been. A first probe jittered the action
history while leaving the lataccel tokens that history had produced untouched, and reported far larger
effects (+120% at amplitude 0.05, +507% at 0.10) -- but that context is impossible, and the model may
simply be uncertain about it. The table above is closed-loop and self-consistent, and the effect
survives at roughly a quarter of the size.

## But the lever is already pulled

Smoothing beyond what `cnn_v4` already does:

| arm | E[Var] | implied floor | mean abs du | track | total |
|---|---|---|---|---|---|
| `cnn_v4` raw | 0.001075 | 28.60 | 0.01281 | 28.63 | 50.30 |
| EMA alpha=0.3 | 0.001026 | 27.30 | 0.01185 | 30.89 | 51.18 |
| EMA alpha=0.6 | 0.001033 | 27.48 | 0.01136 | 41.34 | 61.65 |
| EMA alpha=0.85 | 0.001022 | 27.20 | 0.01166 | 111.01 | 130.41 |

Variance bottoms out around 0.00102 -- only **4-5% below** where `cnn_v4` already sits -- and the
floor with it, 28.60 -> 27.20. That 1.4 points is the entire remaining prize, and collecting it costs
80+ points of tracking, because past a point EMA stops removing roughness (mean abs du barely falls,
0.01281 -> 0.01166) and only adds lag.

So the policy trained end-to-end on the true cost has already found the smooth end of this trade
without being told the mechanism existed.

## Two smaller observations from the same probe

**The current action barely moves the next emission.** Sweeping the current action over +-0.6 at fixed
context moves the conditional mean by ~0.01 per unit -- essentially nothing. That is the dead time,
and it agrees with the measured impulse response (`h[0] ~ -0.03`, response appearing at steps 1-2).
Control here acts through the history, never through the current step.

**The plant is most certain about in-distribution actions.** In the same sweep, conditional variance
is *minimised* exactly at the action the policy actually chose (0.001263) and rises ~19% for any
offset in either direction. An action inconsistent with the recent history makes the model less sure
what happens next -- which is the same phenomenon as the jitter result, seen one step at a time.

## What this changes

The floor numbers stay usable, but their meaning is now precise: `31.24`, and the per-segment values in
`segfloor.py`, are the floor **for a smooth controller**, which is the only regime worth measuring in.
They are not a property of the plant alone, and a rough controller does not get to claim them.

It also retires the idea that any of the remaining gap is recoverable by shaping the noise. It is
controllable, we checked, and `cnn_v4` is within 5% of the minimum the plant will give.
