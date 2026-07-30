"""pid_smooth plus two ways of not commanding faster than the plant can respond.

The concern this addresses: the controller issues a new action every 100 ms, but the plant needs
~5 steps to deliver. So corrections are stacked on top of commands still in flight, and the loop
keeps correcting an error that is already being fixed. That is the classic dead-time
over-correction, and it is why every good controller here runs ~2x detuned.

Two strategies, selected by `mode`:

  'latch'  the naive version: latch the reference and HOLD it until the plant is within `tol` of it,
           then latch the next one. Prediction: this should HURT. Holding makes the reference a
           staircase, and step changes inject exactly the high-frequency content the jerk term
           punishes at 10000x. This project already rejected intermittent control on that argument --
           with quadratic jerk, smearing a change of D over n steps costs D^2/n, so continuous is
           optimal, and unlike a human the benchmark charges no per-action fee. Tested anyway.

  'rate'   the principled version, a reference governor: do not hold the reference, RATE-LIMIT it to
           `max_rate` per step so it never demands more than the plant can deliver. Same goal, but it
           stays continuous, so it does not manufacture jerk. For scale: the plant's hard clamp is
           MAX_ACC_DELTA = 0.5/step, while the jerk the cost actually wants is ~0.045/step, so the
           useful range is between those.

  'off'    reproduces pid_smooth exactly.

Rate limiting is applied to the REFERENCE the loop chases, after Tikhonov smoothing and after the
lookahead index, so it composes with both rather than replacing them.
"""
import numpy as np
from . import BaseController
from .ff_pi import _thomas
from .pid_phys import BASIS


class Controller(BaseController):
    def __init__(self, p=0.195, i=0.100, d=-0.053, i_clip=1e9, basis='t90', scale=0.4,
                 lam=2.0, past=20, mode='off', tol=0.05, max_rate=0.10):
        self.p, self.i, self.d, self.i_clip = p, i, d, i_clip
        self.a, self.b = BASIS[basis]
        self.basis, self.scale, self.lam, self.past = basis, float(scale), float(lam), int(past)
        self.mode, self.tol, self.max_rate = mode, float(tol), float(max_rate)
        self.integ = 0.0
        self.prev_err = 0.0
        self.hist = []
        self.latched = None          # currently held reference ('latch' mode)
        self.prev_ref = None         # last emitted reference ('rate' mode)

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

        if self.mode == 'latch':
            # hold the latched reference until the plant gets within tol of it
            if self.latched is None or abs(current_lataccel - self.latched) <= self.tol:
                self.latched = ref
            ref = self.latched
        elif self.mode == 'rate':
            # reference governor: never move the reference faster than max_rate per step
            if self.prev_ref is None:
                self.prev_ref = current_lataccel
            ref = float(np.clip(ref, self.prev_ref - self.max_rate, self.prev_ref + self.max_rate))
            self.prev_ref = ref

        e = ref - current_lataccel
        self.integ = float(np.clip(self.integ + e, -self.i_clip, self.i_clip))
        de = e - self.prev_err
        self.prev_err = e
        return self.p * e + self.i * self.integ + self.d * de
