"""pid_smooth + pending-response awareness: stop integrating error that in-flight commands will fix.

The problem this targets. A new action is issued every 100 ms but the plant needs ~5 steps to deliver
it, so corrections stack on top of commands still in flight and the loop keeps correcting an error
that is already being fixed. That compounding is why dead-time loops must be detuned.

How this differs from the Smith predictor (`pid_lag.py`, which lost ~16 points). That corrected the
WHOLE error signal by substituting a model prediction into the feedback, and a Smith predictor is
known to degrade disturbance rejection -- fatal here, where the disturbance is a random walk with
lag-1 autocorrelation 0.98. The compounding described above is specifically an INTEGRATOR problem, so
`pred_i` applies the correction to the integral path only and leaves proportional/derivative on the
raw measured error. Fast disturbance rejection is preserved; windup on already-corrected error is not.
`pred_p` exposes the same correction on the proportional path, so the two can be separated -- if the
Smith diagnosis is right, `pred_p` should hurt while `pred_i` helps.

Precedent: conditional integration is the single biggest classical win in this project
(`ff_pi_rl2`, freezing the integrator while the plant's rate clamp binds, verified ~2 points).

Pending response is driven by action INCREMENTS, not levels -- a constant action has nothing pending.
From the measured impulse response H = [0, 0.02, 0.08, 0.22, 0.38, 0.30] (unit DC gain), the
cumulative delivered fraction is [0, 0.02, 0.10, 0.32, 0.70, 1.00], so an increment issued m steps
ago still has `1 - cum[m]` outstanding:

    pending = G(v) * sum_m (1 - cum[m]) * du[t-m]      m = 1..4

`pred_i = pred_p = 0` reproduces pid_smooth exactly.
"""
import numpy as np
from pathlib import Path
from . import BaseController
from .ff_pi import _thomas
from .pid_phys import BASIS

GAIN_FIT = np.load(Path(__file__).resolve().parent.parent / 'gain_fit.npy')
H = np.array([0.00, 0.02, 0.08, 0.22, 0.38, 0.30])
CUM = np.cumsum(H)                                  # delivered fraction after m steps
PEND = np.clip(1.0 - CUM[1:5], 0.0, 1.0)            # outstanding fraction for du[t-1..t-4]


class Controller(BaseController):
    def __init__(self, p=0.195, i=0.100, d=-0.053, i_clip=1e9, basis='t90', scale=0.4,
                 lam=2.0, past=20, pred_i=0.0, pred_p=0.0, gain_scale=1.0):
        self.p, self.i, self.d, self.i_clip = p, i, d, i_clip
        self.a, self.b = BASIS[basis]
        self.basis, self.scale, self.lam, self.past = basis, float(scale), float(lam), int(past)
        self.pred_i, self.pred_p, self.gain_scale = float(pred_i), float(pred_p), float(gain_scale)
        self.integ = 0.0
        self.prev_err = 0.0
        self.hist = []
        self.du = [0.0] * len(PEND)      # recent action increments, most recent first
        self.prev_u = 0.0

    def G(self, v):
        return float(np.clip(self.gain_scale * np.polyval(GAIN_FIT, v), 0.3, 4.0))

    def steps(self, v):
        return max(0.0, self.scale * (self.a + self.b * float(v)))

    def smooth(self, target, future):
        tau = np.array(self.hist[-self.past:] + [target] + list(future), dtype=np.float64)
        k0 = len(self.hist[-self.past:])
        self.hist.append(target)
        if self.lam <= 0.0:
            return tau, k0
        n = len(tau); lam = self.lam
        diag = np.full(n, 1 + 2 * lam); diag[0] = 1 + lam; diag[-1] = 1 + lam
        return _thomas(np.full(n, -lam), diag, np.full(n, -lam), tau.copy()), k0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        future = future_plan.lataccel if future_plan.lataccel else []
        c, k0 = self.smooth(target_lataccel, future)
        x = float(np.clip(k0 + self.steps(state.v_ego), 0, len(c) - 1))
        j0 = int(np.floor(x)); j1 = min(j0 + 1, len(c) - 1); f = x - j0
        ref = (1.0 - f) * c[j0] + f * c[j1]

        # lataccel still owed by increments already issued
        pending = self.G(state.v_ego) * float(np.dot(PEND, self.du))

        e = ref - current_lataccel
        e_i = e - self.pred_i * pending          # integral path: discount what is already coming
        e_p = e - self.pred_p * pending          # proportional path: same, for separation
        self.integ = float(np.clip(self.integ + e_i, -self.i_clip, self.i_clip))
        de = e - self.prev_err
        self.prev_err = e
        u = self.p * e_p + self.i * self.integ + self.d * de

        self.du = [u - self.prev_u] + self.du[:-1]
        self.prev_u = u
        return u
