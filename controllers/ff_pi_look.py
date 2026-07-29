"""ff_pi_rl2 with a lookahead on the FEEDBACK error as well as the feedforward.

`ff_pi` already anticipates in its feedforward — it inverts the plant against `c[k0 + lead]`. But its
PI feedback compares against `c[k0]`, the smoothed reference at *now*, so the feedback path has no
anticipation at all. On a plant with 2-5 steps of response delay that asks the integrator to correct
an error the action cannot affect until after it has grown.

Evidence this is worth trying: applying exactly this change to a bare PID was worth 26% — stock PID
112.19 -> 82.99 on held-out with a constant 2-step lookahead and gains untouched. That is far larger
than anything a Smith predictor achieved (it lost 16 points), because shifting the REFERENCE leaves
the feedback signal as measured lataccel and so preserves disturbance rejection, whereas the Smith
predictor substitutes a model prediction and destroys it. On a plant whose disturbance is a random
walk with lag-1 autocorrelation 0.98, that distinction decides the result.

`fb_look` is the lookahead in steps applied to the feedback reference, fractionally interpolated so
the tuner sees a smooth objective. `fb_look=0` reproduces ff_pi_rl2 bit-for-bit.

Kept deliberately as a single constant: on the PID, joint tuning of a speed/acceleration schedule
(k = k0 + kv*v/30 + ka*a) recovered nothing over a constant (83.65 vs 82.99) and drove k0 to zero
with kv ~ 2.25, i.e. it rediscovered "about 2" the long way round. That matches the impulse-response
measurement, where bulk response timing is speed-INVARIANT at ~5 steps and only the onset moves.
"""
import numpy as np
from .ff_pi_rl2 import Controller as _RL2


class Controller(_RL2):
    def __init__(self, fb_look=0.0, **kw):
        super().__init__(**kw)
        self.fb_look = float(fb_look)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # replicate ff_pi_rl2.update, changing only the reference the PI error is measured against
        from tinyphysics import MAX_ACC_DELTA
        future = future_plan.lataccel if future_plan.lataccel else []
        c, k0 = self.smooth(target_lataccel, future)
        i1 = min(k0 + self.lead, len(c) - 1)

        # feedback reference: c[k0 + fb_look], fractionally interpolated
        x = float(np.clip(k0 + self.fb_look, 0, len(c) - 1))
        j0 = int(np.floor(x)); j1 = min(j0 + 1, len(c) - 1); f = x - j0
        desired = (1.0 - f) * c[j0] + f * c[j1]

        sat = (self.prev_lat is not None
               and abs(current_lataccel - self.prev_lat) >= 0.99 * MAX_ACC_DELTA)
        self.prev_lat = current_lataccel
        if sat:
            self.frozen = self.hold
        elif self.frozen > 0:
            self.frozen -= 1

        ff = (c[i1] - state.roll_lataccel) / self.G(state.v_ego)
        e = desired - current_lataccel
        if self.frozen > 0:
            self.integ *= (1.0 - self.bleed)
        else:
            self.integ = np.clip(self.integ + e, -self.i_clip, self.i_clip)
        fb = self.kp * e + self.ki * self.integ
        return ff + fb
