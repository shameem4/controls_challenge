"""ff_pi_boot detuning its feedforward when the PREVIEW shows a large-|tau| corner coming.

Third gate tried for the same measured mechanism (see FINDINGS_FFPI_HARD.md): the inversion
`ff = (c_lead - roll)/G(v)` over-commands on tight slow corners. The detune value is real and
transfers -- on hard segments of the tuning split it was never fit on, gain_scale 1.79 -> 2.1 is
worth ~21% -- but it only works when applied from the START of the segment. Reactive gates fail
because by the time the rate clamp binds the over-command has already happened:

    gate on log J* (sigmoid, anticipatory but blunt)   every config worse; +0.46 at best
    gate on clamp binding, hold=3..8 (reactive)        every config worse; +0.95 at best
    same, latching for the rest of the segment         monotonically worse with hold; +1.38 at hold=25,
                                                       and better on only 6 of the 23 segments it fired on

So the gate must be PREDICTIVE and PRECISE. Peak |tau| over the preview separates far better than J*
does -- 3.78 on the hard segments against 0.72 elsewhere -- and it is available at every step.
Blended smoothly rather than switched, because jerk is charged at 10000x.

RESULT. Selected on the tuning split ALL[3000:3800], then validated on pristine ALL[5000:8000]
(n=3000, never used for selection):

    tuning  n=800    ff_pi_boot 50.178 -> 48.262   mean -1.917 [-4.82, -0.08]   fires on  56/800
    PRISTINE n=3000  ff_pi_boot 53.248 -> 50.742   mean -2.506 [-3.62, -1.54]   fires on 207/3000
                     p99        253.9  -> 173.8

The CI excludes zero on both splits and the effect replicates in size. It is purely a tail fix: the
median delta is 0.000 because the gate fires on only 6.9% of segments, and on those it is close to a
coin flip per segment (better on 111 of 207) while cutting p99 by 32%. That is the expected shape for
a mechanism that removes rare large over-commands rather than improving nominal tracking -- but it is
also the shape tail noise takes, which is why it was checked on 3000 held-out segments rather than
promoted off the tuning split.

A second config (gs_hard=2.1, tau 3.0-4.0) also validates at -1.865 [-2.86, -1.00], firing on only
112/3000. The two bracket the same effect at different gate widths.

`gs_hard=None` reduces to ff_pi_boot exactly.
"""
import numpy as np
from .ff_pi_boot import Controller as _Boot


class Controller(_Boot):
    def __init__(self, gs_hard=2.1, tau_lo=2.5, tau_hi=3.5, **kw):
        kw.setdefault('boot', 0.005)
        super().__init__(**kw)
        self.gs_norm = self.gain_scale
        self.gs_hard = self.gs_norm if gs_hard is None else float(gs_hard)
        self.tau_lo, self.tau_hi = float(tau_lo), float(tau_hi)
        self.mu_last = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        fut = future_plan.lataccel if future_plan.lataccel else []
        if len(fut):
            peak = float(np.abs(np.asarray(fut, dtype=np.float64)).max())
            peak = max(peak, abs(float(target_lataccel)))
            self.mu_last = float(np.clip((peak - self.tau_lo) / max(self.tau_hi - self.tau_lo, 1e-6), 0.0, 1.0))
        self.gain_scale = self.gs_norm + (self.gs_hard - self.gs_norm) * self.mu_last
        return super().update(target_lataccel, current_lataccel, state, future_plan)
