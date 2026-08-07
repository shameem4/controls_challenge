"""ff_pi_tau with a notch in the FEEDBACK path, targeting the 0.5-1.0 Hz resonance.

`FINDINGS_SPECTRUM.md` decomposed the cost by frequency (exactly -- the cost is a frequency-weighted
objective, so Parseval applies) and found **58% of the gap to the seed-aware optimum sits in
0.5-1.0 Hz**, a band where the reference has almost no content: 99.55% of target power is below 1 Hz
and the 0.5-1.0 band holds 0.0022 against 0.2481 below 0.1 Hz.

So the loop is producing error at a frequency nothing is asking it to follow. That is a loop-shaping
defect rather than broadband noise, and it has a physical identity: with dead time L ~ 0.25 s
(`FINDINGS_SYSID.md`), the textbook closed-loop resonance of a dead-time-dominated loop sits at
f ~ 1/(4L) = 1 Hz. The loop rings at its own natural frequency; the seed-aware optimum, which
pre-compensates, does not.

The intervention is to attenuate loop gain in that band. A second-order RBJ notch on the feedback
term does exactly that and leaves the feedforward -- which carries the low-frequency tracking --
untouched.

    alpha = sin(w0) / (2Q),   w0 = 2 pi f0 / fs,   fs = 10 Hz
    b = [1, -2cos(w0), 1] / (1 + alpha)
    a = [1, -2cos(w0), (1 - alpha)] / (1 + alpha)

`depth` blends between the raw and notched feedback, so the notch can be applied partially. A notch
adds phase lag around the stop band, and on a dead-time-dominated loop phase is the scarce resource --
so a partial notch may beat a full one, and `depth=0` reproduces `ff_pi_tau` bit-for-bit.
"""
import numpy as np
from .ff_pi_tau import Controller as _FFTau
from tinyphysics import MAX_ACC_DELTA

FS = 10.0          # 1 / DEL_T


class _Biquad:
    """Direct-form-I RBJ notch. State is per-controller, so per-segment."""

    def __init__(self, f0, Q):
        w0 = 2.0 * np.pi * f0 / FS
        alpha = np.sin(w0) / (2.0 * Q)
        a0 = 1.0 + alpha
        self.b = np.array([1.0, -2.0 * np.cos(w0), 1.0]) / a0
        self.a = np.array([-2.0 * np.cos(w0), 1.0 - alpha]) / a0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def __call__(self, x):
        y = (self.b[0] * x + self.b[1] * self.x1 + self.b[2] * self.x2
             - self.a[0] * self.y1 - self.a[1] * self.y2)
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return float(y)


class Controller(_FFTau):
    def __init__(self, f0=0.7, Q=2.0, depth=0.0, **kw):
        super().__init__(**kw)
        self.depth = float(depth)
        self.nf = _Biquad(float(f0), float(Q)) if self.depth > 0 else None

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        if self.nf is None:
            return super().update(target_lataccel, current_lataccel, state, future_plan)
        # The parent computes the smoothed reference inline and does not retain it, so the feedback
        # term cannot be recovered after the fact. The body below mirrors ff_pi_boot.update with
        # ff_pi_tau's tau-gated gain, and differs in ONE place: the notch on fb. `depth=0` takes the
        # branch above and reproduces the parent bit-for-bit.
        future = future_plan.lataccel if future_plan.lataccel else []
        if len(future):
            peak = max(float(np.abs(np.asarray(future, dtype=np.float64)).max()),
                       abs(float(target_lataccel)))
            self.mu_last = float(np.clip((peak - self.tau_lo) / max(self.tau_hi - self.tau_lo, 1e-6),
                                         0.0, 1.0))
        self.gain_scale = self.gs_norm + (self.gs_hard - self.gs_norm) * self.mu_last
        c, k0 = self.smooth(target_lataccel, future)
        desired = c[k0]
        i1 = min(k0 + self.lead, len(c) - 1)
        sat = (self.prev_lat is not None
               and abs(current_lataccel - self.prev_lat) >= 0.99 * MAX_ACC_DELTA)
        self.prev_lat = current_lataccel
        if sat:
            self.frozen = self.hold
        elif self.frozen > 0:
            self.frozen -= 1
        ff = (c[i1] - state.roll_lataccel) / self.G(state.v_ego)
        if self.boot > 0.0 and self.ki > 1e-9:
            u_true = (c[i1] - state.roll_lataccel) / self.G_true(state.v_ego)
            self.integ += self.boot * ((u_true - ff) / self.ki - self.integ)
        e = desired - current_lataccel
        if self.frozen > 0:
            self.integ *= (1.0 - self.bleed)
        else:
            self.integ = float(np.clip(self.integ + e, -self.i_clip, self.i_clip))
        fb = self.kp * e + self.ki * self.integ
        fb = (1.0 - self.depth) * fb + self.depth * self.nf(fb)
        return ff + fb
