"""ff_pi_boot with the plant gain scheduled on OPERATING LATACCEL as well as speed.

Motivation is a measured decomposition. Against `ff_pi_boot`, `cnn_v2`'s 3.36-point advantage on 500
pristine segments breaks down as:

    regime               % steps   d(lataccel)  d(jerk)     net    share
    |tau| < 0.2           72.17%      -0.86      +0.38     -0.48    14%
    0.2 - 0.5             12.88%      -0.44      +0.31     -0.13     4%
    0.5 - 1.0              8.57%      -0.74      +0.23     -0.51    15%
    1.0 - 2.0              5.70%      -0.60      +0.15     -0.45    13%
    |tau| > 2.0            0.67%      -1.20      -0.59     -1.79    53%

53% of the learned controller's edge comes from 0.67% of the steps -- hard corners -- where it beats
the classical controller on BOTH terms at once. Everywhere else there is a tracking/jerk tradeoff.

The likely reason: `ff_pi_boot` inverts the plant with `gain_fit.npy`, a function of SPEED ALONE,
while system identification (`FINDINGS_SYSID.md`) measured the gain rising sharply with operating
lataccel at fixed speed (v=22, du=0.30):

    c0 = -2.0 -> 2.418     c0 = -1.0 -> 1.637     c0 = 0.0 -> 1.415
    c0 = +1.0 -> 1.936     c0 = +2.0 -> 7.348  (excluded: |lataccel| > 2 is 0.8% of data and the
                                                plant is extrapolating there, so this is not physics)

So the classical controller is mis-modelled precisely where its losses are largest. This schedules
the gain on the operating point:

    G_eff = G(v) * clip(1 + beta*|c|, 1, cap)

fitted from the trustworthy points (|c| <= 1) which imply beta ~ 0.26.

Scope, stated honestly: this models THIS PLANT, not a vehicle. The operating-point dependence is
almost certainly an artifact of a learned model extrapolating in a sparse region -- a real car does
not change steering gain by 40% between straight and a 1 m/s^2 corner. It is legitimate for the
benchmark (which scores against this plant) and should not be read as vehicle physics.

`beta=0` reproduces ff_pi_boot exactly.
"""
import numpy as np
from .ff_pi_boot import Controller as _Boot
from tinyphysics import MAX_ACC_DELTA


class Controller(_Boot):
    def __init__(self, beta=0.0, cap=2.2, asym=0.0, op='cur', **kw):
        super().__init__(**kw)
        self.beta = float(beta)      # gain rise per unit |lataccel|
        self.cap = float(cap)        # ceiling on the multiplier; the c0=+2 measurement is untrusted
        self.asym = float(asym)      # optional left/right term, measured at 16% (see ff_pi_asym)
        self.op = op                 # 'cur' = current lataccel, 'ref' = the commanded reference

    def _mult(self, c_op, want):
        # clip BOTH ends: with beta<0 an unbounded lower end drives the multiplier to zero at large
        # |lataccel| and the feedforward divides by ~0 (observed: cost 194.9 at beta=-0.15, then a
        # NaN blow-up at beta=-0.25).
        m = float(np.clip(1.0 + self.beta * abs(c_op), 0.25, self.cap))
        if self.asym:
            m *= (1.0 + self.asym) if want > 0 else (1.0 - self.asym)
        return m

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        future = future_plan.lataccel if future_plan.lataccel else []
        c, k0 = self.smooth(target_lataccel, future)
        desired = c[k0]
        i1 = min(k0 + self.lead, len(c) - 1)

        sat = (self.prev_lat is not None
               and abs(current_lataccel - self.prev_lat) >= 0.99 * MAX_ACC_DELTA)
        self.prev_lat = current_lataccel
        if sat:
            self.frozen = self.hold
        elif self.frozen > 0:
            self.frozen -= 1

        want = c[i1] - state.roll_lataccel
        c_op = current_lataccel if self.op == 'cur' else c[i1]
        mult = self._mult(c_op, want)
        ff = want / (self.G(state.v_ego) * mult)

        if self.boot > 0.0 and self.ki > 1e-9:
            u_true = want / (self.G_true(state.v_ego) * mult)
            integ_target = (u_true - ff) / self.ki
            self.integ += self.boot * (integ_target - self.integ)

        e = desired - current_lataccel
        if self.frozen > 0:
            self.integ *= (1.0 - self.bleed)
        else:
            self.integ = np.clip(self.integ + e, -self.i_clip, self.i_clip)
        fb = self.kp * e + self.ki * self.integ
        return ff + fb
