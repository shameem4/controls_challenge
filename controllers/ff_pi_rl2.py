"""ff_pi_tuned plus rate-limit anti-windup. Best classical controller here: 52.30 vs 54.56.

The plant clamps its own lataccel change to MAX_ACC_DELTA = 0.5 per step. That clamp sits about 11x
above normal operation -- a typical jerk cost of ~20 implies RMS lataccel change ~0.045/step -- and
because the benchmark charges jerk quadratically at 10000x, a single saturated step costs 25-64x a
normal one. Across all 20,000 segments, 626 (3.13%) saturate at least once, averaging 6.8 saturated
steps; they cost 300.9 against 46.9 for clean segments (6.4x) and hold 17.2% of ALL cost.

While the clamp binds, the plant cannot respond at any steer, but a PI integrator keeps accumulating
error it cannot act on -- so when the clamp releases, the stored command overshoots, producing a
burst of both tracking and jerk cost. The remedy is conditional integration: stop integrating while
the plant is saturated. Note `i_clip` does NOT address this; it is a fixed magnitude clamp and is
already at its optimum. What was missing was making the clamp *conditional*.

Saturation is observable at runtime with no segment identification: the controller sees
current_lataccel every step, so |lataccel[t] - lataccel[t-1]| >= MAX_ACC_DELTA means the clamp bound.

`hold` keeps the freeze active for that many steps after saturation is last seen. hold=3 is optimal,
and 3 steps is exactly the plant's dead time as measured independently from its impulse response --
the freeze must persist as long as the command keeps arriving. hold=0 disables the mechanism and
reproduces ff_pi_tuned bit-for-bit.

Verified (data/SYNTHETIC holds 20,000 segments; the repo's quoted "full 5000" metric is ALL[:5000]):

                                n     ff_pi_tuned  ff_pi_rl2   delta    bootstrap 95% CI
    ALL[:5000] (repo basis)  5000          54.555     52.297  -2.258   [-3.236, -1.350]
    ALL[5000:] (never eval) 15000          54.987     53.101  -1.886   [-2.346, -1.483]
    ALL                     20000          54.879     52.900  -1.979   [-2.396, -1.567]

The effect replicates on 15,000 segments never evaluated elsewhere in this repo. The claim rests on
a bootstrap CI and a distribution-free sign test (389 improved / 231 worsened of 620 activated,
p=2.3e-10) rather than a t-test, because the paired deltas are heavy-tailed. Segments where the
mechanism never fires are bit-identical to ff_pi_tuned, so this is a targeted fix rather than a
retuning: it costs nothing on the 96.9% of segments that never touch the clamp.

`bleed` optionally decays the integral while frozen (closer to back-calculation anti-windup) and is
off by default -- it was worth ~1% on a screening subset, inside the noise.
"""
import numpy as np
from .ff_pi import Controller as _FFPI
from .ff_pi_tuned import PARAMS
from tinyphysics import MAX_ACC_DELTA


class Controller(_FFPI):
    def __init__(self, hold=3, bleed=0.0, **kw):
        p = dict(PARAMS); p.update(kw)
        super().__init__(**p)
        self.hold = int(hold)
        self.bleed = float(bleed)
        self.prev_lat = None
        self.frozen = 0          # steps of freeze remaining

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        future = future_plan.lataccel if future_plan.lataccel else []
        c, k0 = self.smooth(target_lataccel, future)
        desired = c[k0]
        i1 = min(k0 + self.lead, len(c) - 1)

        # did the plant's rate clamp bind on the previous step?
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
            self.integ *= (1.0 - self.bleed)      # hold (and optionally bleed off) the integral
        else:
            self.integ = np.clip(self.integ + e, -self.i_clip, self.i_clip)
        fb = self.kp * e + self.ki * self.integ
        return ff + fb
