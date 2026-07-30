"""ff_pi_rl2 + bootstrapped integrator, targeting the RESIDUAL the detuned feedforward leaves.

On the stock PID, anchoring the integrator to a model estimate was worth -38% (110.756 -> 68.412).
That controller has no feedforward, so its integrator must supply the ENTIRE steady-state steering
offset and the bootstrap target is simply `u_model/ki`.

`ff_pi_rl2` is different. Its feedforward already supplies the offset -- but deliberately detuned by
1.79x (it divides by ~2.6 where the measured plant gain is ~1.5), so it systematically UNDER-commands
and the integrator exists to make up the difference. The right bootstrap target is therefore that
residual, not the whole command:

    u_true       = (ref - roll) / G_true(v)      # measured physical gain, scale 1.0
    ff           = (ref - roll) / G(v)           # what the feedforward emits (detuned, scale 1.79)
    integ_target = (u_true - ff) / ki            # steady-state residual the integrator must carry

so the integrator is anchored to the value it *should* settle at rather than discovering it by
accumulating error.

Whether this transfers is genuinely open. It is not obviously redundant the way lookahead was --
three separate anticipation mechanisms came back null on `ff_pi_rl2` because it already anticipates,
whereas this changes how the INTEGRATOR is anchored. Against it: `ff_pi_rl2`'s gains were tuned WITH
the integrator discovering that residual, so `gain_scale` and `ki` are already co-adapted to it.

`boot=0` reproduces ff_pi_rl2 bit-for-bit.
"""
import numpy as np
from pathlib import Path
from .ff_pi_rl2 import Controller as _RL2
from tinyphysics import MAX_ACC_DELTA

GAIN_FIT = np.load(Path(__file__).resolve().parent.parent / 'gain_fit.npy')


class Controller(_RL2):
    def __init__(self, boot=0.0, true_scale=1.0, **kw):
        super().__init__(**kw)
        self.boot = float(boot)
        self.true_scale = float(true_scale)     # 1.0 = the measured physical gain

    def G_true(self, v):
        """Same gain polynomial as G(), at the measured scale instead of the detuned one."""
        return float(np.clip(self.true_scale * np.polyval(GAIN_FIT, v), 0.3, 4.0))

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        future = future_plan.lataccel if future_plan.lataccel else []
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
            integ_target = (u_true - ff) / self.ki       # residual the detuned ff leaves behind
            self.integ += self.boot * (integ_target - self.integ)

        e = desired - current_lataccel
        if self.frozen > 0:
            self.integ *= (1.0 - self.bleed)
        else:
            self.integ = np.clip(self.integ + e, -self.i_clip, self.i_clip)
        fb = self.kp * e + self.ki * self.integ
        return ff + fb
