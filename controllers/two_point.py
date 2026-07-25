"""Salvucci & Gray (2004) two-point visual control model of steering, adapted to this plant.

Original law (generalised discrete form, Peng et al. arXiv:2406.03622 Eq. 15):

    delta(k) = delta(k-1) + sum_i k_n*phi(k-i) + sum_i k_f*Omega(k-i) + k_i*Ts*phi(k)

  phi   -- NEAR-point angle  (~6.2 m ahead, ~0.37 s): lane position / stabilisation
  Omega -- FAR-point angle   (~0.9 s ahead):          upcoming curvature / anticipation

Two structural features that distinguish it from our `ff_pi`:
  * it is INCREMENTAL -- the human commands a *change* in steering, not an absolute angle;
  * near and far enter with separate gains, the far point supplying anticipation that the near
    point cannot (Salvucci & Gray's core argument for needing two points rather than one).

Adaptation: we track lateral acceleration rather than visual angles, so
  phi   -> immediate tracking error            (target_now - current_lataccel)
  Omega -> anticipated demand at the far point (target_far  - current_lataccel)
Both are roll-compensated, and the increment is scaled by 1/G(v) because this plant's
steer->lataccel gain rises with speed (measured 1.9 -> 2.9), which the constant-gain visual
model has no analogue for.
"""
import os
import numpy as np
from pathlib import Path
from . import BaseController

_ROOT = Path(__file__).resolve().parent.parent
GAIN_FIT = np.load(_ROOT / 'gain_fit.npy')
DT = 0.1


def _f(name, default):
    return float(os.environ.get(name, default))


class Controller(BaseController):
    def __init__(self, k_n=None, k_f=None, k_i=None, far=None, near=None,
                 gain_scale=None, roll_ff=None, i_clip=5.0):
        self.k_n = _f('TP_KN', 0.35) if k_n is None else k_n
        self.k_f = _f('TP_KF', 0.25) if k_f is None else k_f
        self.k_i = _f('TP_KI', 0.60) if k_i is None else k_i
        self.far = int(_f('TP_FAR', 9)) if far is None else far      # ~0.9 s, per the model
        self.near = int(_f('TP_NEAR', 3)) if near is None else near  # ~0.3 s
        self.gs = _f('TP_GS', 1.3) if gain_scale is None else gain_scale
        self.roll_ff = _f('TP_ROLLFF', 1.0) if roll_ff is None else roll_ff
        self.delta = 0.0          # the running steering command (incremental law)
        self.pn = 0.0             # phi(k-1)
        self.pf = 0.0             # Omega(k-1)
        self.integ = 0.0
        self.i_clip = i_clip
        self.first = True

    def _G(self, v):
        return float(np.clip(self.gs * np.polyval(GAIN_FIT, v), 0.3, 4.0))

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        fl = future_plan.lataccel
        fr = future_plan.roll_lataccel
        n = len(fl)

        def ahead(i, arr, fallback):
            return arr[min(i, n - 1)] if n else fallback

        # roll-compensated demands at the near and far points
        tgt_near = ahead(self.near, fl, target_lataccel) - ahead(self.near, fr, state.roll_lataccel) * self.roll_ff
        tgt_far = ahead(self.far, fl, target_lataccel) - ahead(self.far, fr, state.roll_lataccel) * self.roll_ff
        cur = current_lataccel - state.roll_lataccel * self.roll_ff

        phi = tgt_near - cur          # near-point error: lane position / stabilisation
        omega = tgt_far - cur         # far-point error: curvature anticipation
        if self.first:
            self.pn, self.pf, self.first = phi, omega, False
            self.delta = (target_lataccel - state.roll_lataccel) / self._G(state.v_ego)

        self.integ = float(np.clip(self.integ + phi, -self.i_clip, self.i_clip))

        # incremental law, scaled into steer units by the (speed-dependent) plant gain
        inc = (self.k_n * (phi + self.pn) + self.k_f * (omega + self.pf)
               + self.k_i * DT * self.integ) / self._G(state.v_ego)
        self.delta = float(np.clip(self.delta + inc, -2.0, 2.0))
        self.pn, self.pf = phi, omega
        return self.delta
