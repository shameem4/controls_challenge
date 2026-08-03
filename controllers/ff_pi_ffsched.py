"""ff_pi_boot with the FEEDFORWARD INVERSION gain scheduled on causal difficulty.

Mechanism (measured, see FINDINGS_FFPI_HARD.md). ff_pi_boot inverts the plant as
`ff = (c_lead - roll) / G(v)` with `G(v) = gain_scale * poly(v)`. On tight slow corners -- peak
|tau| 3.78 vs 0.72, v 14.1 vs 24.3 m/s -- that inversion OVER-commands and drives the plant into its
MAX_ACC_DELTA rate clamp (30% of those segments saturate, against 2% elsewhere). 25 such segments
carry 92% of ff_pi_boot's entire gap to cnn_v2, while on the other 700 ff_pi_boot beats the CNN.

This is consistent with the system-ID finding that the plant's gain depends on the OPERATING POINT,
not just on speed: the true gain is higher at large lateral acceleration than poly(v) predicts, so
1/G(v) over-commands exactly where |tau| is large.

NOT the same as the rejected |lataccel| gain schedule -- that scheduled the FEEDBACK gains and was
harmful in both directions. This scheduls the feedforward inversion, and `lam` with it, since
reference smoothing was a second independent lever on the same segments.

Difficulty membership is the same causal signal as `pid_fuzzy`: a sigmoid on log J* over the preview,
blended smoothly because a hard switch changes the command discontinuously and jerk is charged at
10000x. Defaults reduce to ff_pi_boot exactly.
"""
import numpy as np
from .ff_pi_boot import Controller as _Boot
from .pid_fuzzy import jstar


class Controller(_Boot):
    def __init__(self, gs_hard=None, lam_hard=None, centre=1.0, width=0.8, **kw):
        kw.setdefault('boot', 0.005)
        super().__init__(**kw)
        self.gs_easy = self.gain_scale
        self.lam_easy = self.lam
        self.gs_hard = self.gs_easy if gs_hard is None else float(gs_hard)
        self.lam_hard = self.lam_easy if lam_hard is None else float(lam_hard)
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
        self.gain_scale = self.gs_easy + (self.gs_hard - self.gs_easy) * mu
        self.lam = self.lam_easy + (self.lam_hard - self.lam_easy) * mu
        return super().update(target_lataccel, current_lataccel, state, future_plan)
