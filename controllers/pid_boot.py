"""pid_smooth + a bootstrapped integrator: give the integral path a model-based head start.

The problem. A feedback-only PID has no way to know how much steering a new target needs, so the
required steady-state offset must be DISCOVERED by accumulating error. A trace through a real corner
shows the cost: error stays at +0.13 to +0.41 for seven consecutive steps while steering creeps
1.129 -> 1.146. In real driving you do not integrate your way into a corner -- you put in roughly the
right wheel immediately and then trim.

The idea. The plant model already says what steering a target needs: `u_model = (ref - roll)/G(v)`.
Nudge the integrator toward the value that would produce it, `u_model/ki`, instead of waiting for
error to accumulate there.

Two knobs, so the two mechanisms can be separated:

  boot  per-step blend of the integrator toward `u_model/ki`. boot=0 is plain pid_smooth; boot=1
        pins the integrator to the model every step, which is pure feedforward with no integral
        memory left. The interesting region is small boot -- a soft model prior that speeds up
        transients while leaving the integrator free to track drift.
  ffw   a PARALLEL feedforward path of weight `ffw * u_model`, i.e. the conventional alternative,
        for comparison. ffw=1 is essentially what ff_pi does.

Two things this measures. First, whether bootstrapping costs jerk: the jerk penalty is on LATACCEL
change, and the plant spreads a steering step over ~5 steps behind a rate clamp, so a step of du
produces roughly G*du/3 lataccel per step at its steepest -- under the ~0.045 the cost wants for
du <~ 0.09. Small bootstraps should be nearly free. Second, whether continuous bootstrapping wrecks
drift rejection: on this plant the disturbance is a random walk (lag-1 autocorrelation 0.98) and
integral action is the load-bearing element, so pinning the integrator should hurt badly. Measured
rather than assumed.

`boot=0, ffw=0` reproduces pid_smooth exactly.
"""
import numpy as np
from pathlib import Path
from . import BaseController
from .ff_pi import _thomas
from .pid_phys import BASIS

GAIN_FIT = np.load(Path(__file__).resolve().parent.parent / 'gain_fit.npy')


class Controller(BaseController):
    def __init__(self, p=0.195, i=0.100, d=-0.053, i_clip=1e9, basis='t90', scale=0.4,
                 lam=2.0, past=20, boot=0.0, ffw=0.0, gain_scale=1.0,
                 gate='always', gate_thr=0.06):
        self.p, self.i, self.d, self.i_clip = p, i, d, i_clip
        self.a, self.b = BASIS[basis]
        self.basis, self.scale, self.lam, self.past = basis, float(scale), float(lam), int(past)
        self.boot, self.ffw, self.gain_scale = float(boot), float(ffw), float(gain_scale)
        # `boot` is a blend rate, so 0.02 is a 50-step (5 s) time constant -- a slow standing bias,
        # not a jump-start. `gate` discriminates the two readings: if the value lies in fast
        # correction on target changes, 'transient' keeps it; if it lies in a slow anti-drift prior,
        # 'steady' keeps it and 'transient' destroys it.
        self.gate, self.gate_thr = gate, float(gate_thr)
        self.prev_ref = None
        self.integ = 0.0
        self.prev_err = 0.0
        self.hist = []

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

        # steering the model says this reference needs
        u_model = (ref - state.roll_lataccel) / self.G(state.v_ego)

        dref = 0.0 if self.prev_ref is None else abs(ref - self.prev_ref)
        self.prev_ref = ref
        active = (self.gate == 'always'
                  or (self.gate == 'transient' and dref >= self.gate_thr)
                  or (self.gate == 'steady' and dref < self.gate_thr))
        if self.boot > 0.0 and self.i > 1e-9 and active:
            # nudge the integrator toward the value that would produce u_model
            self.integ += self.boot * (u_model / self.i - self.integ)

        e = ref - current_lataccel
        self.integ = float(np.clip(self.integ + e, -self.i_clip, self.i_clip))
        de = e - self.prev_err
        self.prev_err = e
        return self.p * e + self.i * self.integ + self.d * de + self.ffw * u_model
