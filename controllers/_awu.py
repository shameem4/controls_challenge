"""Conditional anti-windup as a mixin, so it can be layered on any pid_boot-derived controller.

Freeze the integrator while the plant's MAX_ACC_DELTA rate clamp is binding, and hold the freeze for
the plant's ~3-step dead time so it does not immediately re-arm. Same mechanism as `ff_pi_rl2`, the
largest classical win in this project.

The integrator is snapshotted before the parent's `update` and restored after, rather than
reimplementing the parent's lookahead/smoothing/bootstrap body -- that stays exactly in sync with
whatever base class it is mixed into. The emitted command is corrected by `i * (kept - accumulated)`
so it is consistent with the integrator that is actually retained.

`awu=False` reproduces the base class bit-for-bit.
"""
from tinyphysics import MAX_ACC_DELTA


class AWUMixin:
    def __init__(self, awu=True, hold=3, bleed=0.0, **kw):
        super().__init__(**kw)
        self.awu = bool(awu)
        self.hold = int(hold)
        self.bleed = float(bleed)
        self.prev_lat = None
        self.frozen = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        if not self.awu:
            return super().update(target_lataccel, current_lataccel, state, future_plan)
        sat = (self.prev_lat is not None
               and abs(current_lataccel - self.prev_lat) >= 0.99 * MAX_ACC_DELTA)
        self.prev_lat = current_lataccel
        if sat:
            self.frozen = self.hold
        elif self.frozen > 0:
            self.frozen -= 1
        before = self.integ
        u = super().update(target_lataccel, current_lataccel, state, future_plan)
        if self.frozen > 0:
            after = self.integ
            self.integ = before * (1.0 - self.bleed)
            u += self.i * (self.integ - after)
        return u
