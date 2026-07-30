"""PID whose lookahead is DERIVED from the measured plant response time, scheduled on velocity.

The premise: for a control action at speed v it takes x(v) steps for the plant to reach the
commanded lataccel, so the error should be measured against the target at t + x(v) rather than at t.
Unlike `pid_look` (where the lookahead was a free tuned parameter), x(v) here comes from measurement.

Measured step response to a +1 m/s^2 request (on-manifold baseline, expected mode, 384 segments,
dt = 0.1 s), time to reach a fraction of final value:

    v band (m/s)      t10%    t50%    t90%
      0.0 - 18.1     400ms   500ms   700ms      (4, 5, 7 steps)
     18.1 - 26.8     400ms   500ms   600ms      (4, 5, 6 steps)
     26.8 - 30.9     300ms   400ms   500ms      (3, 4, 5 steps)
     30.9 - 37.2     300ms   400ms   500ms      (3, 4, 5 steps)

Linear fits in v give the three `basis` options below. `scale` multiplies the whole thing, so
scale=1.0 is the raw physical response time and sweeping it answers a question the tuned version
could not: what FRACTION of the plant's response time is the right amount to anticipate?

There is a known tension worth stating up front. A free tuned lookahead optimised to ~2 steps
(112.19 -> 82.99 on held-out), and a constant 5 scored 109.05 -- far worse. So the optimum is well
short of the 5-7 step response time. Anticipation is not simply "aim where the target will be when
the action lands": looking too far ahead commits to a target that has not arrived, and the cost of
that premature commitment outweighs the phase advance. This controller measures where the trade-off
actually sits, in units of the physical response time.

`scale=0` reduces to the stock PID exactly. Defaults are the verified-best configuration
(basis='t90', scale=0.4): 81.132 on the clean split ALL[5000:6000] vs stock PID's 114.645.
"""
import numpy as np
from . import BaseController

# linear fits of steps-to-reach vs speed, from the measurements in the docstring
BASIS = {
    't10': (4.36, -0.040),      # onset
    't50': (5.36, -0.040),      # half response
    't90': (7.70, -0.080),      # settled
}


class Controller(BaseController):
    def __init__(self, p=0.195, i=0.100, d=-0.053, i_clip=1e9, basis='t90', scale=0.4):
        self.p, self.i, self.d, self.i_clip = p, i, d, i_clip
        self.a, self.b = BASIS[basis]
        self.scale = float(scale)
        self.basis = basis
        self.integ = 0.0
        self.prev_err = 0.0

    def steps(self, v):
        """Measured steps for the plant to respond at this speed, scaled."""
        return max(0.0, self.scale * (self.a + self.b * float(v)))

    def _ref(self, target_lataccel, state, future_plan):
        fut = future_plan.lataccel
        if not fut:
            return target_lataccel
        k = min(self.steps(state.v_ego), len(fut) - 1e-6)
        if k <= 0.0:
            return target_lataccel
        x = k - 1.0                                  # future_plan[0] is one step ahead
        if x <= 0.0:
            return (1.0 + x) * fut[0] + (-x) * target_lataccel
        i0 = int(np.floor(x)); i1 = min(i0 + 1, len(fut) - 1); f = x - i0
        return (1.0 - f) * fut[i0] + f * fut[i1]

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        e = self._ref(target_lataccel, state, future_plan) - current_lataccel
        self.integ = float(np.clip(self.integ + e, -self.i_clip, self.i_clip))
        de = e - self.prev_err
        self.prev_err = e
        return self.p * e + self.i * self.integ + self.d * de
