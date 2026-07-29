"""PID whose error is measured against a FUTURE target, with the lookahead scheduled on speed.

Motivation. The plant's response to a steer command is delayed -- measured impulse response onset is
2-4 steps (speed dependent) with the bulk at ~5 -- so an error computed against the target *now* is
asking the controller to correct something its action cannot affect until later. Aiming at where the
target will be when the action lands supplies phase advance.

Why this should behave differently from the Smith predictor (`pid_lag.py`, which lost ~16 points):
that one substitutes a MODEL PREDICTION into the feedback signal, and a Smith predictor is well
known to degrade disturbance rejection -- fatal on a plant whose disturbance is a random walk with
lag-1 autocorrelation 0.98. This changes only the REFERENCE. Feedback is still measured lataccel, so
integral action keeps rejecting drift exactly as before; the loop simply aims further ahead.

Lookahead in steps:

    k(v, a) = k0 + kv * (v / 30) + ka * a        clipped to the available preview

kept FRACTIONAL and linearly interpolated between preview samples, so the tuner sees a smooth
objective rather than an integer cliff. `k0=0, kv=0, ka=0` reduces to the stock PID exactly.

The speed term has measured support: response onset falls from ~4 steps at low speed to ~2 at high
(correlation -0.98 with speed), consistent with lataccel ~ v^2 * curvature. Note the *bulk* response
timing is speed-INVARIANT at ~5 steps, so the speed term may well tune to zero -- that is a real
possible outcome, not a failure of the parameterisation.
"""
import numpy as np
from . import BaseController


class Controller(BaseController):
    def __init__(self, p=0.195, i=0.100, d=-0.053, i_clip=1e9, k0=0.0, kv=0.0, ka=0.0):
        self.p, self.i, self.d, self.i_clip = p, i, d, i_clip
        self.k0, self.kv, self.ka = float(k0), float(kv), float(ka)
        self.integ = 0.0
        self.prev_err = 0.0

    def _look(self, target_lataccel, state, future_plan):
        """Fractionally-interpolated future target at k(v, a) steps ahead."""
        fut = future_plan.lataccel
        if not fut:
            return target_lataccel
        k = self.k0 + self.kv * (state.v_ego / 30.0) + self.ka * state.a_ego
        k = float(np.clip(k, 0.0, len(fut) - 1e-6))
        if k <= 0.0:
            return target_lataccel
        # index 0 of future_plan is one step ahead, so k=1 -> fut[0]
        x = k - 1.0
        if x <= 0.0:                      # blend between "now" and the first preview sample
            return (1.0 + x) * fut[0] + (-x) * target_lataccel
        i0 = int(np.floor(x))
        i1 = min(i0 + 1, len(fut) - 1)
        f = x - i0
        return (1.0 - f) * fut[i0] + f * fut[i1]

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        ref = self._look(target_lataccel, state, future_plan)
        e = ref - current_lataccel
        self.integ = float(np.clip(self.integ + e, -self.i_clip, self.i_clip))
        de = e - self.prev_err
        self.prev_err = e
        return self.p * e + self.i * self.integ + self.d * de
