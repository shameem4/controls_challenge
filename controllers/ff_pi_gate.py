"""ff_pi_traj with the trajectory blend GATED on where jerk actually dominates the local cost.

`FINDINGS_FF_TRAJ.md` measured the trade precisely: trajectory feedback buys jerk with tracking at an
exchange rate of 0.57 (cost units of jerk saved per cost unit of tracking spent), where break-even is
1.0. Applied uniformly it therefore loses. But 0.57 is an AVERAGE over every step of every segment,
and the cost is not uniformly composed -- some stretches are jerk-dominated and others are
tracking-dominated. If the trajectory term is spent only where jerk is the larger share, the
realised exchange rate on the steps that actually receive it should be better than the average.

The gate is causal and uses the cost's own weights. Per step the two terms contribute

    5000 * e^2        (tracking)          10000 * (du)^2      (jerk)

so the local jerk share, on exponentially-weighted recent history, is

    share = 10000 * jerk_ema / (5000 * track_ema + 10000 * jerk_ema)

and the blend opens linearly across a band:

    w = w_max * clip((share - lo) / (hi - lo), 0, 1)

`w_max = 0.0` reproduces `ff_pi_boot` bit-for-bit and is the identity gate.

WHAT WOULD FALSIFY THE IDEA. If `share` does not actually discriminate -- if the steps where jerk
dominates are not the steps where trajectory feedback helps -- the gate degenerates into a noisier
version of a constant `w` and lands on the same curve. That is the null this arm is testing against,
and it is a real possibility: `FINDINGS_SPECTRUM.md` found the cost gap concentrated in a narrow
0.5-1.0 Hz band, which is a FREQUENCY property, not something a local time-domain ratio need track.
"""
import numpy as np
from .ff_pi_traj import Controller as _Traj

W_TRACK, W_JERK = 5000.0, 10000.0


class Controller(_Traj):
    def __init__(self, w_max=1.0, lo=0.5, hi=0.9, ema=0.90, **kw):
        kw['w'] = 1.0 if w_max > 0 else 0.0      # parent's w is overridden per step below
        super().__init__(**kw)
        self.w_max, self.lo, self.hi = float(w_max), float(lo), float(hi)
        self.ema = float(ema)
        self.track_ema = 0.0
        self.jerk_ema = 0.0
        self.u_prev = None
        self.share = 0.0
        self.e_last = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        if self.w_max <= 0.0:
            return super().update(target_lataccel, current_lataccel, state, future_plan)
        # Gate on the PREVIOUS step's realised composition -- strictly causal.
        self.w = self.w_max * float(np.clip((self.share - self.lo) / max(self.hi - self.lo, 1e-6),
                                            0.0, 1.0))
        u = super().update(target_lataccel, current_lataccel, state, future_plan)

        # The error the feedback loop actually tracks -- the smoothed reference, not the raw target.
        # `ff_pi_traj` records it during update(). Using the raw target here charged the deliberate,
        # cost-optimal smoothing deviation to the tracking side of the ratio, understating the jerk
        # share and so opening the gate in the wrong places.
        e = self.e_last
        du = 0.0 if self.u_prev is None else (u - self.u_prev)
        self.u_prev = u
        a = self.ema
        self.track_ema = a * self.track_ema + (1 - a) * e * e
        self.jerk_ema = a * self.jerk_ema + (1 - a) * du * du
        den = W_TRACK * self.track_ema + W_JERK * self.jerk_ema
        self.share = (W_JERK * self.jerk_ema / den) if den > 1e-12 else 0.0
        return u
