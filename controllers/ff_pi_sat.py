"""ff_pi_boot that weakens its feedforward REACTIVELY, while the plant's rate clamp is binding.

Why reactive. The feedforward inversion `ff = (c_lead - roll)/G(v)` over-commands on tight slow
corners and drives the plant into MAX_ACC_DELTA. Weakening it there is worth ~21% on those segments,
and that value transfers to hard segments it was not fit on (tuning-split worst-25: 245.27 -> 193.43
at gain_scale 1.79 -> 2.1; the 32 saturating segments: 178.89 -> 142.09).

But scheduling it on the PREVIEW fails. A sigmoid on log J* fires on hundreds of segments, and
weakening the feedforward where it was not over-driving costs +4.0 on the other 775 to win 52 on 25 --
every configuration of `ff_pi_ffsched` came out worse than baseline on the tuning split.

Saturation is observable directly: `|lataccel[t] - lataccel[t-1]| >= 0.99 * MAX_ACC_DELTA` means the
clamp bound on the last step. So detune the inversion only while that is true, held for the plant's
~3-step dead time. Same anticipatory-vs-reactive trade that `pid_fawu` settled on the pid side.

`gs_sat=None` reduces to ff_pi_boot exactly.
"""
import numpy as np
from .ff_pi_boot import Controller as _Boot
from tinyphysics import MAX_ACC_DELTA


class Controller(_Boot):
    def __init__(self, gs_sat=None, hold=3, **kw):
        kw.setdefault('boot', 0.005)
        super().__init__(**kw)
        self.gs_norm = self.gain_scale
        self.gs_sat = self.gs_norm if gs_sat is None else float(gs_sat)
        self.hold = int(hold)
        # NOT `prev_lat`/`frozen`: ff_pi_boot already uses those names for its inherited
        # anti-windup. Overwriting prev_lat before calling the parent made its saturation test
        # compare the current lataccel to itself, silently disabling anti-windup (+0.6 on the
        # tuning split). The identity gate caught it.
        self._plat = None
        self._froz = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        sat = (self._plat is not None
               and abs(current_lataccel - self._plat) >= 0.99 * MAX_ACC_DELTA)
        self._plat = current_lataccel
        if sat:
            self._froz = self.hold
        elif self._froz > 0:
            self._froz -= 1
        self.gain_scale = self.gs_sat if self._froz > 0 else self.gs_norm
        return super().update(target_lataccel, current_lataccel, state, future_plan)
