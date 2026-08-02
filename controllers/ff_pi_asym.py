"""ff_pi_boot with a DIRECTION-DEPENDENT plant gain.

System identification (`sysid.py`, FINDINGS_SYSID.md) measured a left/right asymmetry the `G(v)`
schedule does not model. At the nominal operating point (v=22, roll=0, c0=0), stepping the steer
command by +-0.6 and averaging 96 noise realisations over 4 seeds:

    du = -0.60   DC gain 1.390 +- 0.020
    du = +0.60   DC gain 1.614 +- 0.010

A 16% difference at roughly 20 sigma, reproduced at c0=1.0 (1.578 vs 2.232) and at low speed. A
positive steer command produces more lataccel than an equal negative one. `gain_fit.npy` is a
function of speed alone, so every controller here -- `ff_pi`, `ff_pi_boot`, the DMC teacher, the
`PMG` gain prior -- inverts the plant with a gain that is ~8% wrong in one direction and ~8% wrong
the other way.

This matters because the regime is the common one: median |target lataccel| is 0.073 and p90 is
0.720, so almost all driving happens near the operating point where the asymmetry was measured.
(The much larger gain excursions seen at |lataccel| > 2 are NOT modelled here -- only 0.8% of data
lives there and the plant is a learned model extrapolating, so those numbers are suspect.)

Correction: the controller inverts the plant as `u = (desired - roll) / G`. If the true gain is
higher for positive commands, a positive command needs LESS steer, i.e. a LARGER effective G:

    G_eff = G(v) * (1 + asym)   when (desired - roll) > 0
    G_eff = G(v) * (1 - asym)   otherwise

The measured ratio 1.614/1.390 = 1.16 implies asym ~ 0.074. Both the feedforward and the bootstrap
anchor use it, since both invert the same plant.

`asym=0` reproduces ff_pi_boot exactly.
"""
import numpy as np
from .ff_pi_boot import Controller as _Boot
from tinyphysics import MAX_ACC_DELTA


class Controller(_Boot):
    def __init__(self, asym=0.0, **kw):
        super().__init__(**kw)
        self.asym = float(asym)

    def _dir(self, x):
        """Multiplier on G for a command in direction sign(x)."""
        return (1.0 + self.asym) if x > 0 else (1.0 - self.asym)

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
        d = self._dir(want)
        ff = want / (self.G(state.v_ego) * d)

        if self.boot > 0.0 and self.ki > 1e-9:
            u_true = want / (self.G_true(state.v_ego) * d)
            integ_target = (u_true - ff) / self.ki
            self.integ += self.boot * (integ_target - self.integ)

        e = desired - current_lataccel
        if self.frozen > 0:
            self.integ *= (1.0 - self.bleed)
        else:
            self.integ = np.clip(self.integ + e, -self.i_clip, self.i_clip)
        fb = self.kp * e + self.ki * self.integ
        return ff + fb
