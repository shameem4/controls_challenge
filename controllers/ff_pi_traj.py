"""ff_pi_boot with the FEEDBACK loop closed on trajectory error instead of lataccel error.

`FINDINGS_TRAJ_PID.md` put trajectory-error control on stock `pid` and it lost badly: two extra
integrators in the error path, on a plant where dead time already makes phase margin the binding
constraint. But on stock `pid` the feedback loop carries the ENTIRE tracking burden, so any phase
it spends comes straight out of tracking.

`ff_pi` is structurally different. Its inverse-plant feedforward on a cost-optimal smoothed
reference supplies the bulk of the command, and the PI exists only to clean up the residual drift.
Feedforward costs no phase margin at all -- it is open loop. So the argument for retrying here is
concrete: the trajectory loop only has to correct residuals, and its lag applies to a much smaller
signal.

Two things change versus `pid_traj`:

  * The trajectory error accumulates against the SMOOTHED reference `c[k0]` that the feedback term
    actually tracks, not the raw target. Integrating against the raw target would charge the loop
    for the smoothing deviation that the cost deliberately wants.
  * `H` defaults to 0. The feedforward already looks ahead, and this repo has a standing pattern of
    added anticipation coming back null on top of `ff_pi_rl2` precisely because it anticipates
    already (see the `ff_pi_boot` docstring). `H > 0` remains available to test, not assumed.

Otherwise identical to `pid_traj`: leaky double integration of the heading-error rate, as in
`drive_template.html:laneOffsets`, folded into a virtual error with the units of lataccel.

    psi <- psi * decay + (-e / v) * dt
    y   <- y   * decay + v * psi * dt
    e_traj = -(k_y * y_pred + k_psi * v * psi)
    e_fb   = (1 - w) * e + w * e_traj

The DC gain of that virtual error is `k_y * tau^2 + k_psi * tau`, which is what the first sweep in
`FINDINGS_TRAJ_PID.md` got wrong by 3-22x. Defaults sit at 0.525 with tau = 3.

`w = 0.0` reproduces `ff_pi_boot` bit-for-bit and is the identity gate.
"""
import numpy as np
from .ff_pi_boot import Controller as _Boot
from tinyphysics import MAX_ACC_DELTA

DT = 0.1


class Controller(_Boot):
    def __init__(self, w=1.0, tau=3.0, H=0, k_y=0.005, k_psi=0.16, **kw):
        kw.setdefault('boot', 0.005)
        super().__init__(**kw)
        self.w = float(w)
        self.decay = float(np.exp(-DT / tau)) if tau > 0 else 0.0
        self.H = int(H)
        self.k_y, self.k_psi = float(k_y), float(k_psi)
        self.psi = 0.0          # heading error, rad
        self.y = 0.0            # lateral offset from the commanded path, m

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # NO early return for w == 0, deliberately. An earlier version delegated to the parent in
        # that case, which left the psi/y recursion below unexecuted. That is harmless at a constant
        # w=0, but `ff_pi_gate` varies w per step and shuts the gate ~97% of the time, so the
        # trajectory state advanced on only 2.9% of steps and was stale whenever the gate opened --
        # the gated arm was not testing the mechanism it claimed to. State must track continuously
        # regardless of how much of it is currently being used.
        #
        # Bit-exactness at w=0 is preserved instead by construction: the blend below is
        # (1-0)*e + 0*x, and 0.0*x is exactly 0.0 for any finite x, so e_fb is e to the last bit.
        # The identity gate covers this.
        #
        # The parent computes the smoothed reference inline and does not retain it, so the feedback
        # error cannot be recovered after the fact. The body below mirrors ff_pi_boot.update and
        # differs in ONE place: the error handed to the PI.
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
            self.integ += self.boot * ((u_true - ff) / self.ki - self.integ)

        e = desired - current_lataccel
        # Exposed so a subclass can gate on the error the loop ACTUALLY tracks. `ff_pi_gate`
        # previously measured its jerk/track ratio against the raw target, which is a different
        # signal: the smoothing deviation is deliberate and cost-optimal, so charging it to the
        # tracking side of the ratio biased the gate.
        self.e_last = e

        v = max(float(state.v_ego), 1e-3)
        self.psi = self.psi * self.decay + (-e / v) * DT
        self.y = self.y * self.decay + v * self.psi * DT
        h = min(self.H, len(c) - 1 - k0)
        if h > 0:
            th = h * DT
            drift = current_lataccel - float(np.mean(c[k0 + 1:k0 + 1 + h]))
            y_pred = self.y + v * self.psi * th + 0.5 * drift * th * th
        else:
            y_pred = self.y
        e_fb = (1.0 - self.w) * e + self.w * (-(self.k_y * y_pred + self.k_psi * v * self.psi))

        if self.frozen > 0:
            self.integ *= (1.0 - self.bleed)
        else:
            self.integ = np.clip(self.integ + e_fb, -self.i_clip, self.i_clip)
        return ff + self.kp * e_fb + self.ki * self.integ
