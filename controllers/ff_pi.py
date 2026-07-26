import numpy as np
from pathlib import Path
from . import BaseController

# smoothing strength from the cost itself: 25.06*(dc)^2 vs 12.5*(c-tau)^2
LAM = 25.06 / 12.5
GAIN_FIT = np.load(Path(__file__).resolve().parent.parent / 'gain_fit.npy')


def _thomas(lower, diag, upper, rhs):
    """Solve a tridiagonal system (Thomas algorithm)."""
    n = len(diag)
    cp = np.empty(n); dp = np.empty(n)
    cp[0] = upper[0] / diag[0]; dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * cp[i - 1]
        cp[i] = upper[i] / m if i < n - 1 else 0.0
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


class Controller(BaseController):
    """Stage 1: inverse-plant feedforward on a cost-optimal smoothed reference,
    plus PI feedback for low-frequency drift."""

    def __init__(self, kp=0.20, ki=0.10, lead=3, past=20, i_clip=5.0, gain_scale=1.3, smooth=True,
                 lam=LAM):
        self.kp, self.ki, self.lead, self.past, self.i_clip = kp, ki, lead, past, i_clip
        self.gain_scale, self.smooth_on = gain_scale, smooth
        self.lam = lam          # smoothing strength; defaults to the cost-derived LAM
        self.integ = 0.0
        self.hist = []  # past targets for a centered smoothing window

    def G(self, v):
        return float(np.clip(self.gain_scale * np.polyval(GAIN_FIT, v), 0.3, 4.0))

    def smooth(self, target, future):
        # window = [past targets ... current ... future targets]; solve (I+LAM D'D)c=tau
        tau = np.array(self.hist[-self.past:] + [target] + list(future))
        k0 = len(self.hist[-self.past:])  # index of "now"
        n = len(tau)
        lam = self.lam
        diag = np.full(n, 1 + 2 * lam); diag[0] = 1 + lam; diag[-1] = 1 + lam
        low = np.full(n, -lam); up = np.full(n, -lam)
        c = _thomas(low, diag, up, tau.copy()) if self.smooth_on else tau
        self.hist.append(target)
        return c, k0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        future = future_plan.lataccel if future_plan.lataccel else []
        c, k0 = self.smooth(target_lataccel, future)
        desired = c[k0]
        lead_idx = min(k0 + self.lead, len(c) - 1)
        # inverse-plant feedforward (roll adds directly, coef ~1.0)
        ff = (c[lead_idx] - state.roll_lataccel) / self.G(state.v_ego)
        # PI feedback on residual
        e = desired - current_lataccel
        self.integ = np.clip(self.integ + e, -self.i_clip, self.i_clip)
        fb = self.kp * e + self.ki * self.integ
        return ff + fb
