"""pid_boot with gains fuzzy-scheduled on a CAUSAL difficulty estimate from the preview.

Not scheduled on speed. Speed barely matters here -- the measured plant gain moves only 1.37 -> 1.83
across 5-34 m/s, and a synthetic-condition sweep found v=8/22/34 within 17% of each other. What does
matter is how hard the upcoming trajectory is to track smoothly.

DIFFICULTY SIGNAL. The benchmark cost is a convex quadratic in the lataccel trajectory alone, so for
any target window the best achievable cost is the closed-form Tikhonov solve

    J*(tau) = min_c [ 5000*mean((c-tau)^2) + 100*mean((dc/dt)^2) ],   (I + 2 D'D) c* = tau

computed in O(n) by the Thomas algorithm over the 50-step preview. That is exact, needs no rollout,
involves no plant or controller, and measures precisely the tracking-vs-jerk conflict: it is large
only when the target has content that cannot be followed without jerk. Crucially it is CAUSAL --
`future_plan` is available at every step -- so it can drive gains in real time.

Measured per-segment, log J* predicts cnn_v2's cost at R^2 = 0.49 (road features alone: 0.29; both
together: 0.36), against a ceiling of 0.83 set by noise-draw variability.

WHY SCHEDULE. Tuning pid_boot separately on the easiest and hardest 300 of 1000 held-out segments
(split by J*, medians 0.79 vs 12.51) gives:

    p=0.145   i:   0.050    0.075    0.100    0.140    0.200
    EASY          52.60    41.13    35.64    32.43   190.53     <- wants i = 0.14
    HARD         182.61   134.19   118.49   212.46  4506.67     <- wants i = 0.10, and 0.14 costs 79%

Both prefer p=0.145; they differ in the INTEGRAL gain, and the penalty is asymmetric -- easy pays
10% at i=0.10 while hard pays 79% at i=0.14. So a fixed gain must sit at the hard optimum and give
up the easy gain, which is what scheduling recovers.

FUZZY BLEND. Two-rule Takagi-Sugeno: a smooth membership on log J* interpolates between an "easy"
and a "hard" gain set. Smoothness matters -- a hard switch changes the command discontinuously and
the jerk term charges lataccel change at 10000x.

    mu   = sigmoid((log J*_preview - centre) / width)
    i(t) = i_easy + (i_hard - i_easy) * mu          (same form for p)

`i_easy == i_hard` reduces to fixed gains, and with p/i at their defaults it reproduces pid_boot.
"""
import numpy as np
from .pid_boot import Controller as _Boot
from .ff_pi import _thomas

LAM = 2.0          # = W_jerk / W_track, the cost-optimal Tikhonov weight
DEL_T = 0.1


def jstar(tau, lam=LAM):
    """Closed-form minimum achievable benchmark cost for this target window. O(n)."""
    n = len(tau)
    if n < 3:
        return 0.0
    d = np.full(n, 1 + 2 * lam); d[0] = 1 + lam; d[-1] = 1 + lam
    c = _thomas(np.full(n, -lam), d, np.full(n, -lam), np.asarray(tau, dtype=np.float64).copy())
    return 5000.0 * np.mean((c - tau) ** 2) + 100.0 * np.mean((np.diff(c) / DEL_T) ** 2)


class Controller(_Boot):
    def __init__(self, p_easy=0.145, p_hard=0.145, i_easy=0.140, i_hard=0.100,
                 centre=1.0, width=0.8, **kw):
        kw.setdefault('boot', 0.02)
        super().__init__(**kw)
        self.p_easy, self.p_hard = float(p_easy), float(p_hard)
        self.i_easy, self.i_hard = float(i_easy), float(i_hard)
        self.centre, self.width = float(centre), float(width)
        self.mu_last = 0.0

    def _mu(self, future):
        """Fuzzy membership in 'hard', from log J* over the preview window."""
        if len(future) < 3:
            return self.mu_last
        j = jstar(np.asarray(future, dtype=np.float64))
        x = (np.log(j + 1e-6) - self.centre) / max(self.width, 1e-6)
        self.mu_last = float(1.0 / (1.0 + np.exp(-x)))
        return self.mu_last

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        future = future_plan.lataccel if future_plan.lataccel else []
        mu = self._mu(future)
        # blend BEFORE the parent runs, so the parent's own p/i are what get used
        self.p = self.p_easy + (self.p_hard - self.p_easy) * mu
        self.i = self.i_easy + (self.i_hard - self.i_easy) * mu
        return super().update(target_lataccel, current_lataccel, state, future_plan)
