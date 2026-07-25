"""Receding-horizon LQ-MPC on the identified LPV-ARX plant model, with offset-free
disturbance estimation.

Plant model (coefficients scheduled on v_ego):
    lat[k] = a(v)*lat[k-1] + b(v)*u[k] + c(v)*roll[k] + d(v) + w
`w` is an estimated disturbance: the measured one-step residual, low-pass filtered and held
constant over the horizon. The plant's stochasticity is a slow drift (lag-1 autocorr 0.98), so a
constant-offset assumption over a ~30-step horizon is a good approximation -- this is what
absorbs most of the model mismatch.

Cost matches the challenge exactly. Per scored step the challenge weights
    50 * lataccel_cost + jerk_cost = 5000*mean((lat-target)^2) + 10000*mean((dlat)^2)
so the jerk:tracking weight ratio is 2:1 in squared units (the same ratio ff_pi's Tikhonov
smoother uses).

Because the model is linear and the cost quadratic, the horizon problem is a convex QP; the
unconstrained optimum is a single HxH linear solve, after which the action is clipped.
"""
import os
import numpy as np
from pathlib import Path
from . import BaseController

_ROOT = Path(__file__).resolve().parent.parent
H = int(os.environ.get('MPC_H', 30))            # horizon
W_TRACK, W_JERK = 5000.0, 10000.0               # exact challenge weights
# Move suppression. The challenge cost penalises only lataccel and jerk, so with a perfect model
# no input penalty is needed -- but our LPV model is ~2x the plant's own predictive floor, and
# without damping the MPC inverts the one-step gain, overshoots, and the loop rings. R_DU is the
# standard robustness knob for MPC under model mismatch.
R_DU = float(os.environ.get('MPC_RDU', 3000.0))     # penalty on u[k]-u[k-1]
R_U = float(os.environ.get('MPC_RU', 0.0))          # penalty on u level
DIST_TAU = float(os.environ.get('MPC_DTAU', 0.0))   # disturbance estimator gain (0 = off)
W_CLIP = float(os.environ.get('MPC_WCLIP', 0.5))    # anti-windup bound on the disturbance estimate
STEER_MIN, STEER_MAX = -2.0, 2.0
BOX = os.environ.get('MPC_BOX', '1') == '1'   # enforce |u|<=2 across the whole plan


class Controller(BaseController):
    def __init__(self, model=None, horizon=None):
        z = np.load(_ROOT / (model or os.environ.get('MPC_MODEL', 'lpv_arx1.npz')))
        self.C, self.V = z['C'], z['V_EDGES']       # rows: [a, b, c, d]
        self.H = horizon or H
        self.prev_lat = None
        self.prev_u = 0.0
        self.w = 0.0                                 # disturbance estimate
        self.plan = None                             # previous plan (warm start)
        self.pred_next = None                        # model's prediction of this step's lataccel

    def _solve_box(self, Hq, g, n, iters=60):
        """min 0.5 u'Hq u + g'u  s.t. |u| <= STEER_MAX, by FISTA projected gradient.
        Dependency-free (a QP solver pulled in scipy and broke the numpy/matplotlib ABI).
        Warm-started from the previous plan, so it converges in few iterations."""
        L = float(np.abs(Hq).sum(1).max()) + 1e-12      # Gershgorin bound on the largest eigenvalue
        step = 1.0 / L
        u = np.zeros(n)
        if self.plan is not None and self.plan.size == n:
            u[:-1] = self.plan[1:]; u[-1] = self.plan[-1]   # shift previous plan
        y = u.copy(); t = 1.0
        for _ in range(iters):
            grad = Hq @ y + g
            u_new = np.clip(y - step * grad, STEER_MIN, STEER_MAX)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            y = u_new + ((t - 1) / t_new) * (u_new - u)
            u, t = u_new, t_new
        return u

    def _coef(self, v):
        i = int(np.clip(np.searchsorted(self.V, v, side='right') - 1, 0, len(self.V) - 2))
        c = self.C[i]
        return c[0], c[1], c[2], c[3]

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # --- offset-free disturbance update: how wrong was last step's one-step prediction? ---
        if self.pred_next is not None:
            # integrating estimator: pred_next already contains w, so this drives w until the
            # model reproduces the measured lataccel
            self.w += DIST_TAU * (current_lataccel - self.pred_next)
            self.w = float(np.clip(self.w, -W_CLIP, W_CLIP))    # anti-windup
        # --- assemble the horizon: future targets/roll/v (edge-padded) ---
        fl = np.asarray(future_plan.lataccel, dtype=np.float64)
        fr = np.asarray(future_plan.roll_lataccel, dtype=np.float64)
        fv = np.asarray(future_plan.v_ego, dtype=np.float64)
        n = self.H

        def pad(arr, head):
            if arr.size >= n:
                return arr[:n]
            if arr.size == 0:
                return np.full(n, head)
            return np.concatenate([arr, np.full(n - arr.size, arr[-1])])

        tgt = pad(fl, target_lataccel)
        roll = pad(fr, state.roll_lataccel)
        vv = pad(fv, state.v_ego)
        # step 0 of the horizon is the action we are about to apply -> use current state
        tgt = np.concatenate([[target_lataccel], tgt[:-1]])
        roll = np.concatenate([[state.roll_lataccel], roll[:-1]])
        vv = np.concatenate([[state.v_ego], vv[:-1]])

        A = np.empty(n); B = np.empty(n); S = np.empty(n)
        for k in range(n):
            a, b, c, d = self._coef(vv[k])
            A[k] = a; B[k] = b
            S[k] = c * roll[k] + d + self.w          # exogenous + disturbance
        # --- free response p and input matrix M (lat = M u + p) ---
        P = np.cumprod(A)                             # P[k] = prod_{i<=k} a_i
        p = P * current_lataccel + P * np.cumsum(S / P)
        M = np.tril(B[None, :] * (P[:, None] / P[None, :]))
        # --- quadratic cost: W_TRACK||lat-tgt||^2 + W_JERK||D lat||^2, D vs previous lataccel ---
        D = np.eye(n) - np.eye(n, k=-1)
        d0 = np.zeros(n); d0[0] = current_lataccel    # first jerk term is lat[0]-current
        MtD = D @ M
        Hq = W_TRACK * (M.T @ M) + W_JERK * (MtD.T @ MtD)
        g = W_TRACK * (M.T @ (p - tgt)) + W_JERK * (MtD.T @ (D @ p - d0))
        # move suppression: R_DU*||Du u - u_prev e0||^2 + R_U*||u||^2
        if R_DU:
            u0 = np.zeros(n); u0[0] = self.prev_u
            Hq += R_DU * (D.T @ D)
            g += R_DU * (D.T @ (-u0))
        if R_U:
            Hq[np.diag_indices(n)] += R_U
        Hq[np.diag_indices(n)] += 1e-6                # numerical guard
        if BOX:
            u = self._solve_box(Hq, g, n)
        else:
            try:
                u = np.linalg.solve(Hq, -g)
            except np.linalg.LinAlgError:
                u = np.zeros(n)
        self.plan = u
        act = float(np.clip(u[0], STEER_MIN, STEER_MAX))
        # --- remember this step's model prediction for the next disturbance update ---
        a0, b0, c0, dd0 = self._coef(state.v_ego)
        self.pred_next = a0 * current_lataccel + b0 * act + c0 * state.roll_lataccel + dd0 + self.w
        self.prev_u = act
        return act
