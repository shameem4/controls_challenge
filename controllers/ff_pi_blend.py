"""Two controllers, blended each step by the blend that minimises the PREDICTED one-step cost.

Rather than gating trajectory feedback on a heuristic (`ff_pi_gate.py`), run both controllers to
completion every step and let the cost function itself choose the mix:

    u_L = ff_pi_boot(...)          lataccel-error feedback, the incumbent
    u_T = ff_pi_traj(w=1, ...)     trajectory-error feedback, smoother but worse-tracking
    u(b) = u_L + b * (u_T - u_L)

The choice of `b` is not a heuristic and not a search. The realised cost per step is

    J = 5000 * e^2 + 10000 * (du)^2

and both terms are quadratic in `b`, because `u(b)` is affine in `b`. With a one-step plant model

    lat_pred(b) = lat + alpha * (G(v) * u(b) + roll - lat)

the predicted error is `e_pred(b) = A - B*b` and the command increment is `du(b) = C + d*b`, where

    d = u_T - u_L
    A = ref_next - lat - alpha * (G * u_L + roll - lat)
    B = alpha * G * d
    C = u_L - u_prev

so J(b) is a scalar quadratic with a closed-form minimum:

    b* = (5000 * A * B - 10000 * C * d) / (5000 * B^2 + 10000 * d^2)

clipped to [b_lo, b_hi]. One divide per step, no iteration. This is a one-step MPC restricted to the
line between two controllers -- the restriction is what makes it a blend rather than the full
trajectory optimisation already explored in `ideal_mpc.py`, and it is a genuine regulariser: the
one-step model is crude, and a 1-D feasible set limits how far a bad prediction can push the command.

`alpha` is the one-step response fraction. `FINDINGS_SYSID.md` measured dead time L ~ 0.25 s and a
sustained-offset DC gain of ~2.4 (`verify_plant.py`); `alpha` is left as a swept parameter rather
than derived, because the closed-loop response is not first-order and the right effective value
depends on where in the band the loop is operating.

`b_fix=0.0` reproduces `ff_pi_boot` bit-for-bit and is the identity gate. `b_fix=1.0` gives pure
trajectory control, matching `ff_pi_traj(w=1)`.

HONEST EXPECTATION. `FINDINGS_FF_TRAJ.md` showed the trajectory arm's shape is inert -- only the
amount blended in mattered. If that holds, the per-step optimiser mostly rediscovers a constant `b`
and lands on the same curve. What would make this different is if `b*` correlates with something
real (corner entry, saturation, the 0.5-1.0 Hz resonance), which is measurable from the realised
`b*` trace and is worth reporting whether or not the total improves.
"""
import numpy as np
from pathlib import Path
from . import BaseController
from .ff_pi_boot import Controller as _Boot
from .ff_pi_traj import Controller as _Traj

W_TRACK, W_JERK = 5000.0, 10000.0
GAIN_FIT = np.load(Path(__file__).resolve().parent.parent / 'gain_fit.npy')


class Controller(BaseController):
    def __init__(self, alpha=0.40, b_lo=0.0, b_hi=1.0, b_fix=None,
                 traj=None, boot=0.005, **kw):
        self.L = _Boot(boot=boot, **kw)
        self.T = _Traj(w=1.0, boot=boot, **(traj or {}), **kw)
        self.alpha = float(alpha)
        self.b_lo, self.b_hi = float(b_lo), float(b_hi)
        self.b_fix = None if b_fix is None else float(b_fix)
        self.u_prev = None
        self.b_hist = []          # realised blend, for diagnosis

    @staticmethod
    def _G(v):
        return float(np.clip(np.polyval(GAIN_FIT, v), 0.3, 4.0))

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        uL = self.L.update(target_lataccel, current_lataccel, state, future_plan)
        if self.b_fix == 0.0:
            self.u_prev = uL
            return uL
        uT = self.T.update(target_lataccel, current_lataccel, state, future_plan)

        d = uT - uL
        if self.b_fix is not None:
            b = self.b_fix
        elif abs(d) < 1e-12:
            b = 0.0
        else:
            fut = future_plan.lataccel if future_plan.lataccel else []
            ref_next = float(fut[0]) if len(fut) else float(target_lataccel)
            G = self._G(state.v_ego)
            A = ref_next - current_lataccel - self.alpha * (
                G * uL + state.roll_lataccel - current_lataccel)
            B = self.alpha * G * d
            C = 0.0 if self.u_prev is None else (uL - self.u_prev)
            den = W_TRACK * B * B + W_JERK * d * d
            b = (W_TRACK * A * B - W_JERK * C * d) / den if den > 1e-12 else 0.0
            b = float(np.clip(b, self.b_lo, self.b_hi))

        self.b_hist.append(b)
        u = uL + b * d
        self.u_prev = u
        return u
