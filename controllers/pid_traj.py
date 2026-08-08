"""PID driven by TRAJECTORY error instead of lataccel error -- lataccel becomes emergent.

Every controller in this repo so far closes the loop on lateral acceleration: the error fed to the
PID is `target_lataccel - current_lataccel`. That is an inner loop. A real driving stack closes an
outer loop on where the car actually IS relative to where the plan says it should be, and lateral
acceleration is merely the actuation that gets it there.

This controller inverts that. It reconstructs the trajectory the same way the visualisation does
(`drive_template.html:laneOffsets`), because the dataset has no road geometry but curvature is
determined exactly by lataccel and speed:

    kappa = a_lat / v^2        heading rate   psi_dot = v * kappa = a_lat / v

so integrating the lataccel ERROR once gives a heading error, and twice gives a lateral offset:

    psi <- psi * decay + ((a_cur - a_tgt) / v) * dt          heading error, rad
    y   <- y   * decay + v * psi * dt                        lateral offset, m

`decay = exp(-dt / tau)` makes both integrators leaky. That is not a cosmetic detail. There is no
position feedback anywhere in this problem -- the target lataccel is a fixed recording, so nothing
ever observes or corrects position -- and a pure double integrator would let any tiny bias run away
without bound. A real planner re-derives the target from lane position every tick and kills such a
bias in about a second; `tau` stands in for that missing outer loop. `FINDINGS_VISUAL_VS_COST.md`
measured the consequence: with tau = 3 s the DC gain from a constant lataccel error to displayed
offset is only ~9.3, so a 0.009 bias shows as 0.085 m rather than the 7 m an open-loop integration
over the window would produce.

The PREDICTED part uses the preview. Holding the current lataccel error over a lookahead of `H`
steps, the offset at the end of the horizon is

    y_pred = y + v * psi * (H dt) + 0.5 * (a_cur - mean(future targets over H)) * (H dt)^2

which is the constant-error extrapolation -- a pure-pursuit lookahead point, not an MPC rollout.
`FINDINGS_PREVIEW.md` found the cost function can only use about 1 s of the 5 s the harness gives,
so H is kept short by default.

The two trajectory terms are folded into a single virtual error with the units of lataccel, so the
parent's PID gains still have meaning:

    e_traj = -(k_y * y_pred + k_psi * v * psi)

`w` blends it against the ordinary lataccel error. **`w = 0.0` reproduces stock `pid` bit-for-bit**
and is the identity gate; `w = 1.0` is the experiment proper, where the PID never sees lataccel
error at all and lataccel is purely emergent from trajectory control.

WHAT THIS COSTS. Position error is the double integral of lataccel error, so `w > 0` inserts two
extra integrators into the error path. The plant is dead-time dominated (L ~ 0.25 s,
`FINDINGS_SYSID.md`) and phase margin is already the binding constraint -- the Bode waterbed result
in `FINDINGS_SPECTRUM.md` is exactly that story. Two integrators spend phase margin freely, so the
expected trade is worse tracking and smoother steering. The cost weights lataccel error directly at
50x, so this is unlikely to win on the metric even if it drives better in the sense a human means.
"""
import numpy as np
from . import BaseController

DT = 0.1


class Controller(BaseController):
  def __init__(self, w=1.0, tau=3.0, H=8, k_y=0.60, k_psi=1.20,
               p=0.195, i=0.100, d=-0.053):
    self.p, self.i, self.d = float(p), float(i), float(d)
    self.w = float(w)
    self.decay = float(np.exp(-DT / tau)) if tau > 0 else 0.0
    self.H = int(H)
    self.k_y, self.k_psi = float(k_y), float(k_psi)
    self.error_integral = 0.0
    self.prev_error = 0.0
    self.psi = 0.0          # heading error, rad
    self.y = 0.0            # lateral offset from the commanded path, m

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    e_lat = target_lataccel - current_lataccel

    if self.w > 0.0:
      v = max(float(state.v_ego), 1e-3)
      # Same recursion as the visualisation, run causally: leaky double integration of the
      # heading-error rate. Order matters -- psi advances first, then y integrates the new psi.
      self.psi = self.psi * self.decay + (-e_lat / v) * DT
      self.y = self.y * self.decay + v * self.psi * DT

      fut = future_plan.lataccel if future_plan.lataccel else []
      h = min(self.H, len(fut))
      if h > 0:
        # Constant-error extrapolation to the lookahead point. `a_cur - mean(future target)` is the
        # error we would keep accumulating if nothing changed.
        th = h * DT
        drift = current_lataccel - float(np.mean(fut[:h]))
        y_pred = self.y + v * self.psi * th + 0.5 * drift * th * th
      else:
        y_pred = self.y

      e_traj = -(self.k_y * y_pred + self.k_psi * v * self.psi)
      error = (1.0 - self.w) * e_lat + self.w * e_traj
    else:
      error = e_lat

    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    return self.p * error + self.i * self.error_integral + self.d * error_diff
