"""pid_w_ff with a SPEED-SCHEDULED averaging kernel instead of the fixed [5,6,7,8] weights.

Upstream averages `[target] + future[0:3]` with weights [5,6,7,8]. That is a smoothing kernel whose
center of mass is 44/26 = 1.692 steps -- i.e. an implicit lookahead, brackets `ff_pi`'s hand-tuned
lead=2 and `pid_phys`'s measured optimum ~2.2. So the weights are an anticipation knob wearing a
smoothing costume, and the question is whether that center should move with speed.

Generalized to a Gaussian kernel over `[target] + future[0:horizon-1]`:

    w_j = exp(-(j - mu)^2 / (2*sigma^2)),   mu(v) = mu0 + mu_v * (v / 30)

`mu_v` is signed, so BOTH directions are reachable and the sign is measured, not assumed. This
matters because the two plausible arguments point opposite ways:

  - measured step response says LESS lead at high speed. t90 to a +1 m/s^2 request is 700 ms below
    18 m/s but 500 ms above 27 m/s, and the fit t90 = 7.70 - 0.080*v (steps) scheduled this way took
    the stock PID 114.645 -> 81.132 on a clean split. That wants mu_v < 0.
  - at high speed the reference itself moves faster (a given curve is traversed in fewer steps), so
    the future arrives sooner in time. That wants mu_v > 0.

The first effect is measured on this plant; the second is a guess about the reference. Sweeping
settles it. Note upstream ALSO already scales its feedforward by 1/max(20, v_ego), so a speed
dependence exists in the ff magnitude -- this adds one in the ff *timing*, which is separate.

`kern='orig'` reproduces upstream bit-for-bit (verified by equality test, not by inspection).
"""
import math
import numpy as np
from .pid_w_ff import Controller as _WFF


class Controller(_WFF):
    def __init__(self, kern='gauss', mu0=1.692, mu_v=0.0, sigma=1.2, horizon=8):
        super().__init__()
        self.kern = kern
        self.mu0, self.mu_v, self.sigma = float(mu0), float(mu_v), float(sigma)
        self.horizon = int(horizon)

    def blend(self, target_lataccel, future, v_ego):
        """Weighted average of [target] + future[0:horizon-1] under the scheduled kernel."""
        if self.kern == 'orig':
            if len(future) >= 3:
                return float(np.average([target_lataccel] + future[0:3], weights=[5, 6, 7, 8]))
            return target_lataccel
        n = min(self.horizon, 1 + len(future))
        if n < 2:
            return target_lataccel
        tau = np.array([target_lataccel] + list(future[:n - 1]), dtype=np.float64)
        j = np.arange(n, dtype=np.float64)
        mu = self.mu0 + self.mu_v * (float(v_ego) / 30.0)
        w = np.exp(-0.5 * ((j - mu) / self.sigma) ** 2)
        return float(np.dot(w, tau) / w.sum())

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer=math.inf):
        # body kept identical to pid_w_ff; only the averaging line is replaced by self.blend()
        self.counter += 1
        if self.counter == 81:
            self.error_integral = 0
            self.prev_error = 0

        target_lataccel = self.blend(target_lataccel, future_plan.lataccel, state.v_ego)

        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        pid_factor = min(1, 1 - (abs(target_lataccel) - 1) * 0.23)
        p = (self.p - abs(state.a_ego) / 10)
        u_pid = (p * error + self.i * self.error_integral + self.d * error_diff) * pid_factor

        steer_accel_target = (target_lataccel - state.roll_lataccel)
        steer_command = (steer_accel_target * self.steer_factor /
                         max(self.steer_sat_v, state.v_ego))
        steer_command = 2 * self.steer_command_sat / (1 + math.exp(-steer_command)) - self.steer_command_sat
        u_ff = 0.8 * steer_command
        return u_pid + u_ff
