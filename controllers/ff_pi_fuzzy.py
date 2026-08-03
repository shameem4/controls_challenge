"""ff_pi_boot with kp/ki fuzzy-scheduled on the causal difficulty estimate J*.

Port of the `pid_fuzzy` mechanism to the best classical controller. Same difficulty signal -- the
closed-form Tikhonov optimum over the 50-step preview, see `pid_fuzzy` for the derivation -- and the
same two-rule Takagi-Sugeno blend on log J*.

One difference matters: ff_pi_boot ALREADY carries the conditional anti-windup it inherits from
`ff_pi_rl2`, and a bounded `i_clip=5.0`. On the pid side the scheduler only became a net win once
anti-windup was stacked underneath it (`pid_fawu`), because the aggressive easy-side integral gain
otherwise wound up against the plant's rate clamp. Here that protection is already present, so the
easy side can be pushed directly.

All four gain arguments default to the base class's tuned values, so `Controller()` reproduces
ff_pi_boot exactly. Note ff_pi_boot's tuned ki is already 0.135 -- near the pid side's EASY optimum,
and bounded by i_clip=3.11 -- so here the schedule has room on the HARD side, not the easy one.
"""
import numpy as np
from .ff_pi_boot import Controller as _Boot
from .pid_fuzzy import jstar


class Controller(_Boot):
    def __init__(self, kp_easy=None, kp_hard=None, ki_easy=None, ki_hard=None,
                 centre=1.0, width=0.8, **kw):
        kw.setdefault('boot', 0.005)
        super().__init__(**kw)
        # Default to whatever the base class was tuned to, so `Controller()` is a no-op. Hardcoding
        # the docstring values here silently overwrote the tuned gains (kp 0.142 -> 0.20,
        # ki 0.135 -> 0.10) and cost 5.3 points before the equality gate caught it.
        self.kp_easy = self.kp if kp_easy is None else float(kp_easy)
        self.kp_hard = self.kp if kp_hard is None else float(kp_hard)
        self.ki_easy = self.ki if ki_easy is None else float(ki_easy)
        self.ki_hard = self.ki if ki_hard is None else float(ki_hard)
        self.centre, self.width = float(centre), float(width)
        self.mu_last = 0.0

    def _mu(self, future):
        if len(future) < 3:
            return self.mu_last
        x = (np.log(jstar(np.asarray(future, dtype=np.float64)) + 1e-6) - self.centre) / max(self.width, 1e-6)
        self.mu_last = float(1.0 / (1.0 + np.exp(-x)))
        return self.mu_last

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        mu = self._mu(future_plan.lataccel if future_plan.lataccel else [])
        self.kp = self.kp_easy + (self.kp_hard - self.kp_easy) * mu
        self.ki = self.ki_easy + (self.ki_hard - self.ki_easy) * mu
        return super().update(target_lataccel, current_lataccel, state, future_plan)
