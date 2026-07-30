"""ff_pi_rl2 with a VELOCITY-SCHEDULED feedforward lead, derived from the measured response time.

`ff_pi` inverts the plant against `c[k0 + lead]` with a constant `lead = 2`. But the plant's response
time varies with speed -- measured step response to a +1 m/s^2 request:

    v band (m/s)      t10%    t50%    t90%
      0.0 - 18.1     400ms   500ms   700ms     (4, 5, 7 steps)
     18.1 - 26.8     400ms   500ms   600ms     (4, 5, 6 steps)
     26.8 - 30.9     300ms   400ms   500ms     (3, 4, 5 steps)
     30.9 - 37.2     300ms   400ms   500ms     (3, 4, 5 steps)

so less anticipation is needed at high speed. Scheduling the lead this way took the stock PID from
114.645 to 81.132 on a clean split (-29%, gains untouched), and the schedule beat a tuned constant by
-4.0. That was on a controller whose ONLY channel is feedback; here the same idea is applied to the
FEEDFORWARD tap of the best classical controller.

This is NOT the `fb_look` experiment that failed. That added lookahead to the FEEDBACK error, which
is redundant with an existing feedforward -- a fine search drove it to 1e-06. This reschedules the
feedforward tap itself, which nothing has tested.

    lead_eff(v) = scale * (a + b*v)        fractionally interpolated into the smoothed reference

`vlead=0` disables the schedule and falls back to the integer `lead`, reproducing ff_pi_rl2 exactly.
Note the PID's optimum was scale ~0.4 of the t90 fit -- about 2.0-2.9 steps, bracketing ff_pi's
hand-tuned constant 2, so the expected gain here is the *shape*, not the magnitude.
"""
import numpy as np
from .ff_pi_rl2 import Controller as _RL2
from .pid_phys import BASIS
from tinyphysics import MAX_ACC_DELTA


class Controller(_RL2):
    def __init__(self, vlead=1.0, basis='t90', scale=0.4, **kw):
        super().__init__(**kw)
        self.vlead = float(vlead)
        self.a, self.b = BASIS[basis]
        self.basis, self.vscale = basis, float(scale)

    def lead_steps(self, v):
        """Velocity-scheduled feedforward lead in steps; falls back to the constant when vlead=0."""
        if self.vlead <= 0.0:
            return float(self.lead)
        sched = self.vscale * (self.a + self.b * float(v))
        # vlead blends between the constant lead and the schedule, so 1.0 is pure schedule
        return max(0.0, (1.0 - self.vlead) * self.lead + self.vlead * sched)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        future = future_plan.lataccel if future_plan.lataccel else []
        c, k0 = self.smooth(target_lataccel, future)
        desired = c[k0]

        # feedforward tap: fractional, velocity-scheduled
        x = float(np.clip(k0 + self.lead_steps(state.v_ego), 0, len(c) - 1))
        j0 = int(np.floor(x)); j1 = min(j0 + 1, len(c) - 1); f = x - j0
        ref_ff = (1.0 - f) * c[j0] + f * c[j1]

        sat = (self.prev_lat is not None
               and abs(current_lataccel - self.prev_lat) >= 0.99 * MAX_ACC_DELTA)
        self.prev_lat = current_lataccel
        if sat:
            self.frozen = self.hold
        elif self.frozen > 0:
            self.frozen -= 1

        ff = (ref_ff - state.roll_lataccel) / self.G(state.v_ego)
        e = desired - current_lataccel
        if self.frozen > 0:
            self.integ *= (1.0 - self.bleed)
        else:
            self.integ = np.clip(self.integ + e, -self.i_clip, self.i_clip)
        fb = self.kp * e + self.ki * self.integ
        return ff + fb
