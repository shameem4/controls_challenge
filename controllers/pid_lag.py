"""PID with a Smith predictor — dead-time compensation using the measured impulse response.

Why. The plant's response to a steer command is delayed: measured impulse response, normalised to
unit DC gain, is roughly

    lag k     0     1     2     3     4     5
    h[k]    0.00  0.02  0.08  0.22  0.38  0.30      (onset ~2 steps, bulk at ~5, peak at 5)

Dead time is what caps loop gain, and it is why every good controller here is heavily detuned —
ff_pi_tuned commands about half the steer that true plant inversion calls for. A plain PID has no
way to know that its correction is already "in flight" but not yet visible, so it keeps correcting
and then overshoots.

The Smith predictor is the textbook fix. Model the plant as `delay * P0`, run the model twice — once
with the delay and once without — and feed the controller

    feedback = y_measured + (y_model_undelayed - y_model_delayed)

so the loop effectively sees the delay-free plant, and the PID can be tuned as if there were no
dead time. The measured residual `y_measured - y_model_delayed` keeps it honest when the model is
wrong, which matters here because the noise is a random walk the model cannot predict.

`smith=0` disables prediction and reduces this to an ordinary PID on the measured lataccel, so the
same file provides the control arm — any difference is dead-time compensation alone. With the
default unbounded `i_clip` it reproduces comma's stock PID bit-for-bit (that controller has no
integral clamp, so a finite default would silently make the control arm a different controller).

Gains MUST be retuned per `smith` setting. A Smith predictor changes the loop's effective dynamics,
so comparing the two at fixed gains measures nothing — the point of removing the delay from the loop
is that you can then afford gains a delayed loop could not.

Model taps are the measured impulse response scaled by `G(v)` (verified accurate to +-9% by
step-response identification). The delay-free model is the same response with its leading dead time
removed, which preserves DC gain.
"""
import numpy as np
from pathlib import Path
from . import BaseController

GAIN_FIT = np.load(Path(__file__).resolve().parent.parent / 'gain_fit.npy')

# measured impulse response, normalised to unit DC gain (sums to 1)
H = np.array([0.00, 0.02, 0.08, 0.22, 0.38, 0.30])
DEAD = 2                                  # leading taps that are ~zero: the transport delay
H0 = H[DEAD:] / H[DEAD:].sum()            # same response with the dead time removed


class Controller(BaseController):
    def __init__(self, p=0.195, i=0.100, d=-0.053, i_clip=1e9, gain_scale=1.0, smith=1.0):
        self.p, self.i, self.d = p, i, d
        self.i_clip, self.gain_scale, self.smith = i_clip, gain_scale, float(smith)
        self.integ = 0.0
        self.prev_err = 0.0
        self.u = [0.0] * len(H)           # past actions, most recent first

    def G(self, v):
        return float(np.clip(self.gain_scale * np.polyval(GAIN_FIT, v), 0.3, 4.0))

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        g = self.G(state.v_ego)
        # model response to the actions already committed, with and without the transport delay
        y_del = g * float(np.dot(H, self.u))
        y_und = g * float(np.dot(H0, self.u[:len(H0)]))
        fb = current_lataccel + self.smith * (y_und - y_del)

        e = target_lataccel - fb
        self.integ = float(np.clip(self.integ + e, -self.i_clip, self.i_clip))
        de = e - self.prev_err
        self.prev_err = e
        out = self.p * e + self.i * self.integ + self.d * de

        self.u = [out] + self.u[:-1]
        return out
