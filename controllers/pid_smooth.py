"""pid_phys + Tikhonov-smoothed target: stop chasing target content that costs more than it saves.

Two goals, per the benchmark's own cost `50*mean((lat-tau)^2)*100 + mean((dlat/dt)^2)*100`:
  * don't chase target detail whose tracking benefit is outweighed by the jerk it induces;
  * reduce near-term jerk directly.

Why Tikhonov rather than a Kalman filter. A Kalman filter is optimal for recovering a state from
NOISY measurements, but the target here is exact -- `future_plan` is the true desired lataccel, so
there is nothing to estimate. What is actually wanted is a tracking-vs-jerk TRADE-OFF, and for this
cost that trade-off has a closed-form optimum: the Whittaker-Tikhonov smoother

    (I + lam * D'D) c = tau        with lam = W_jerk / W_track = 2

which is exactly `ff_pi`'s `c*`. (The two ideas are related -- Tikhonov smoothing is the Kalman
smoother under a random-walk prior -- but here the weights come from the cost function instead of
being assumed.) Measured: tracking `c*` perfectly costs 5.844 against 9.779 for the raw target,
better on 200/200 segments.

Solved causally over a window of [past targets ... now ... available preview] by the Thomas
algorithm, so it uses only information the controller legitimately has. The lookahead then indexes
into the SMOOTHED reference rather than the raw target, so the two mechanisms compose:

    e = c[k0 + k(v)] - lataccel        k(v) from the measured response time (pid_phys)

`lam=0` gives `(I)c = tau`, i.e. no smoothing, and reproduces `pid_phys` exactly.
"""
import numpy as np
from . import BaseController
from .ff_pi import _thomas
from .pid_phys import BASIS


class Controller(BaseController):
    def __init__(self, p=0.195, i=0.100, d=-0.053, i_clip=1e9,
                 basis='t90', scale=0.4, lam=2.0, past=20):
        self.p, self.i, self.d, self.i_clip = p, i, d, i_clip
        self.a, self.b = BASIS[basis]
        self.basis, self.scale, self.lam, self.past = basis, float(scale), float(lam), int(past)
        self.integ = 0.0
        self.prev_err = 0.0
        self.hist = []                      # past targets, for a centred smoothing window

    def steps(self, v):
        return max(0.0, self.scale * (self.a + self.b * float(v)))

    def smooth(self, target, future):
        """(I + lam D'D) c = tau over [past ... now ... future]; returns c and the index of 'now'."""
        tau = np.array(self.hist[-self.past:] + [target] + list(future), dtype=np.float64)
        k0 = len(self.hist[-self.past:])
        self.hist.append(target)
        if self.lam <= 0.0:
            return tau, k0
        n = len(tau)
        lam = self.lam
        diag = np.full(n, 1 + 2 * lam); diag[0] = 1 + lam; diag[-1] = 1 + lam
        low = np.full(n, -lam); up = np.full(n, -lam)
        return _thomas(low, diag, up, tau.copy()), k0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        future = future_plan.lataccel if future_plan.lataccel else []
        c, k0 = self.smooth(target_lataccel, future)
        # lookahead indexes into the SMOOTHED reference, fractionally interpolated
        x = float(np.clip(k0 + self.steps(state.v_ego), 0, len(c) - 1))
        j0 = int(np.floor(x)); j1 = min(j0 + 1, len(c) - 1); f = x - j0
        ref = (1.0 - f) * c[j0] + f * c[j1]

        e = ref - current_lataccel
        self.integ = float(np.clip(self.integ + e, -self.i_clip, self.i_clip))
        de = e - self.prev_err
        self.prev_err = e
        return self.p * e + self.i * self.integ + self.d * de
