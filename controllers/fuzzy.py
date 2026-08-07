"""Takagi-Sugeno fuzzy logic controller: a genuine FLC, not gain scheduling.

Distinct from `pid_fuzzy` / `ff_pi_fuzzy`, which use one fuzzy membership to SCHEDULE a PID's gains.
Here the fuzzy inference system IS the controller: a rule base over (error, error rate) fires 25
rules and their weighted consequents produce the command directly.

STRUCTURE

  inputs      e  = desired - current            normalised by E_SCALE, clipped to [-1, 1]
              de = e - e_prev                   normalised by DE_SCALE
  form='pos'  (default) inputs are (error, INTEGRAL); the surface output IS the command.
  form='vel'  inputs are (error, error rate); the output is du, accumulated.
  sets        5 triangular per input at centres -1, -0.5, 0, +0.5, +1 (NB NS ZE PS PB), width 0.5
  rules       25, the classic skew-symmetric FLC table: the consequent depends on (i-2)+(j-2),
              so error and error-rate contribute symmetrically and the surface is monotone
  consequent  order-0 singletons, shaped by GAMMA (below)
  output      INCREMENTAL: du, accumulated into the command.

WHY POSITIONAL FORM IS THE DEFAULT. The textbook PI-like FLC is velocity form: inputs (e, de),
output du, accumulated. Tuned to ff_pi's exact equivalent gains that reproduces the MEDIAN (46.51 vs
46.09) but the MEAN blows up to 74.69. The reason is that in velocity form the proportional action
also lives inside the accumulator, so when the rate-clamp anti-windup freezes or the clip binds, the
proportional response is permanently lost -- while ff_pi's positional form recomputes `kp*e` every
step. That is the classic velocity-form failure under clamping, and it lands on exactly the hard
segments that drive the mean.

Positional form puts the fuzzy surface over (error, integral) instead. The integrator keeps ff_pi's
semantics exactly -- clipped at i_clip, frozen while the plant's rate clamp binds -- and only the
surface that maps (e, integ) to a command is fuzzy. At GAMMA=1 this reduces algebraically to
`kp*e + ki*integ`, i.e. ff_pi's PI, which is the null hypothesis and the sanity gate.

THE ONE NONLINEARITY THAT MATTERS. With linear consequents a TS system of this shape reduces exactly
to a PI controller, so it could not beat one. GAMMA shapes the consequent surface:

    K_ij = OUT * sign(s) * |s|**GAMMA,   s = ((i-2) + (j-2)) / 4  in [-1, 1]

    GAMMA = 1    linear  -> equivalent to PI (the null hypothesis, and the sanity gate)
    GAMMA < 1    aggressive near zero error, saturating for large error
    GAMMA > 1    gentle near zero error, aggressive for large error

That is the whole point of using a fuzzy system here rather than a PI: a tunable nonlinear control
surface with only four parameters, all interpretable.

MODES
  base='tau' (default) the FULL proven stack -- ff_pi_tau's Tikhonov reference, inverse-plant
                      feedforward with the tau-gated detune, bootstrapped integrator and rate-clamp
                      anti-windup -- with the fuzzy system replacing ONLY the `kp*e + ki*integ`
                      term. This is the fair test: those mechanisms are worth ~4 points together, so
                      a fuzzy controller lacking them is being judged on their absence, not on its
                      own inference.
  base='none'         pure FLC on (e, de) with no feedforward at all -- the classic textbook
                      controller, kept as a reference for what the fuzzy inference does alone.
  fuzzy_off=True      identity gate: reproduces ff_pi_tau bit-for-bit.

Why the fuzzy block is given the feedforward rather than asked to replace it: `ff_pi_look`,
`ff_pi_vlead` and two others were null because a second anticipation mechanism has nothing left to
buy once a feedforward exists. A fuzzy block asked to also do the anticipating would be testing the
wrong thing.
"""
import numpy as np
from . import BaseController
from .ff_pi import Controller as _FFPI, GAIN_FIT, LAM, _thomas
from .ff_pi_tau import Controller as _FFTau
from tinyphysics import MAX_ACC_DELTA

CENTRES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
HALFW = 0.5


def _mu(x):
    """Triangular memberships of a scalar already normalised to [-1, 1]."""
    return np.clip(1.0 - np.abs(x - CENTRES) / HALFW, 0.0, None)


def _table(out, gamma):
    """25 consequent singletons from the skew-symmetric FLC rule table."""
    i = np.arange(5) - 2
    s = (i[:, None] + i[None, :]) / 4.0          # [-1, 1]
    return out * np.sign(s) * np.abs(s) ** gamma


class _Fuzzy:
    def __init__(self, e_scale, de_scale, out, gamma):
        self.e_scale, self.de_scale = float(e_scale), float(de_scale)
        self.K = _table(float(out), float(gamma))
        self.prev_e = None

    def step(self, e):
        de = 0.0 if self.prev_e is None else e - self.prev_e
        self.prev_e = e
        we = _mu(np.clip(e / self.e_scale, -1.0, 1.0))
        wd = _mu(np.clip(de / self.de_scale, -1.0, 1.0))
        W = np.outer(we, wd)
        s = W.sum()
        if s <= 1e-12:
            return 0.0
        return float((W * self.K).sum() / s)


class Controller(_FFTau):
    """Fuzzy inference replacing the PI feedback term, on ff_pi_tau's full front end.

    The update body mirrors ff_pi_boot's exactly -- reference smoothing, tau-gated feedforward,
    bootstrap, rate-clamp anti-windup -- and swaps ONE line: `fb = kp*e + ki*integ` becomes the
    fuzzy inference output. Reimplemented rather than delegated because the parent computes the
    feedback inline; the pieces that are copied are marked, and `fuzzy_off=True` restores the
    parent exactly as an identity gate.
    """

    def __init__(self, e_scale=0.3, de_scale=0.25, out=0.08, gamma=1.0,
                 base='tau', u_clip=3.11, fuzzy_off=False, form='pos', i_scale=3.11, **kw):
        kw.setdefault('boot', 0.005)
        super().__init__(**kw)
        self.fz = _Fuzzy(e_scale, de_scale, out, gamma)
        self.form = form
        self.i_scale = float(i_scale)
        self.base = base
        self.u_clip = float(u_clip)
        self.fuzzy_off = bool(fuzzy_off)
        self.u_fb = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        if self.fuzzy_off:
            return super().update(target_lataccel, current_lataccel, state, future_plan)
        if self.base == 'none':
            e = target_lataccel - current_lataccel
            self.u_fb = float(np.clip(self.u_fb + self.fz.step(e), -self.u_clip, self.u_clip))
            return self.u_fb

        future = future_plan.lataccel if future_plan.lataccel else []
        # --- copied from ff_pi_boot.update: tau-gated feedforward on a smoothed reference ---
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
        # --- end copy ---

        e = desired - current_lataccel
        if self.form == 'vel':
            du = self.fz.step(e)
            if self.frozen > 0:                  # anti-windup: hold the accumulated feedback
                du = 0.0
            self.u_fb = float(np.clip(self.u_fb + du, -self.u_clip, self.u_clip))
            return ff + self.u_fb
        # positional: integrator keeps ff_pi's exact semantics, only the (e, integ) -> u map is fuzzy
        if self.frozen > 0:
            self.integ *= (1.0 - self.bleed)
        else:
            self.integ = float(np.clip(self.integ + e, -self.i_clip, self.i_clip))
        we = _mu(np.clip(e / self.fz.e_scale, -1.0, 1.0))
        wi = _mu(np.clip(self.integ / self.i_scale, -1.0, 1.0))
        W = np.outer(we, wi); sw = W.sum()
        fb = 0.0 if sw <= 1e-12 else float((W * self.fz.K).sum() / sw)
        return ff + fb
