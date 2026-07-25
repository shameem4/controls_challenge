"""Gauss-Newton (successive-linearisation) MPC through the smooth surrogate.

Why not gradient descent: measured, plain Adam cannot solve this plan. Even with a PERFECT model
(the MPC controlling the surrogate itself) and no detuning, 150 Adam iterations reach only ~2256
where an optimal controller should reach ~10-30. The Hessian is severely ill-conditioned -- u[0]
moves the whole trajectory, u[H-1] moves only the last step -- which is precisely why MPC is
normally solved with second-order / SQP methods.

Here each control step:
  1. free-run the surrogate on the current plan  -> lat0
  2. get the exact Jacobian J = d lat / d u by autograd (H backward passes, banded lower-triangular
     because the surrogate reads a 20-step action window)
  3. the challenge cost is quadratic in lat, so the linearised sub-problem is an exact QP in du:
        (Wt J'J + Wj (DJ)'(DJ) + Rdu D'D) du = -(Wt J'e + Wj (DJ)'f + Rdu D'(Du - u_prev))
     solved in closed form (batched HxH solve)
  4. take a damped step, clamp to the steer range, repeat a few times.
"""
import numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment, rollout, cost
from surrogate import Surrogate
from tinyphysics import (CONTEXT_LENGTH as CL, CONTROL_START_IDX, COST_END_IDX,
                         MAX_ACC_DELTA, STEER_RANGE)

DEV = 'cuda'
ALL = sorted(Path('data/SYNTHETIC').iterdir())
W_TRACK, W_JERK = 5000.0, 10000.0


class GaussNewtonMPC:
    def __init__(self, sur, B, dev, H=20, iters=3, r_du=0.0, damp=1.0, ridge=1e-3,
                 clamp='soft'):
        self.sur, self.B, self.dev, self.H = sur, B, dev, H
        self.iters, self.r_du, self.damp, self.ridge = iters, r_du, damp, ridge
        # How the plant's MAX_ACC_DELTA slew clamp is represented while PLANNING.
        # 'hard' reproduces the plant exactly but has ZERO gradient once saturated, which blinds
        # the optimiser: it then drives the plant into a permanent +-MAX_ACC_DELTA square wave and
        # cannot escape (this was the real bug). 'soft' saturates smoothly so gradients survive;
        # 'none' omits it and relies on the jerk penalty to keep the plan feasible.
        self.clamp = clamp
        self.plan = torch.zeros(B, H, device=dev)
        n = H
        self.D = (torch.eye(n, device=dev) - torch.eye(n, device=dev).roll(1, 0)).contiguous()
        self.D[0, -1] = 0.0                     # D lat = [lat0, lat1-lat0, ...]

    def _pad(self, arr, head, n):
        k = arr.shape[1]
        if k >= n:
            return arr[:, :n]
        if k == 0:
            return head[:, None].expand(self.B, n)
        return torch.cat([arr, arr[:, -1:].expand(self.B, n - k)], 1)

    def _freerun(self, u, act_h, lat_h, fr, fv, fa, cur0):
        """Returns lat [B,H] predicted by the surrogate for plan u (differentiable in u)."""
        full_act = torch.cat([act_h, u], 1)
        lat_win, prev, out = lat_h, cur0, []
        for h in range(self.H):
            s = slice(h + 1, h + 1 + CL)
            pred = self.sur(full_act[:, s], fr[:, s], fv[:, s], fa[:, s], lat_win)
            d = pred - prev
            if self.clamp == 'hard':
                d = d.clamp(-MAX_ACC_DELTA, MAX_ACC_DELTA)
            elif self.clamp == 'soft':
                d = MAX_ACC_DELTA * torch.tanh(d / MAX_ACC_DELTA)   # smooth, gradient survives
            pred = prev + d
            out.append(pred)
            lat_win = torch.cat([lat_win[:, 1:], pred[:, None]], 1)
            prev = pred
        return torch.stack(out, 1)

    def __call__(self, ctx):
        t, T, n = ctx['step'], ctx['T'], self.H
        if t < CONTROL_START_IDX:
            return torch.zeros(self.B, device=self.dev)
        tgt = torch.cat([ctx['target'][:, None],
                         self._pad(ctx['fut_lat'], ctx['target'], n)], 1)[:, :n]
        end = min(t + n, T)
        fr = torch.cat([ctx['roll_all'][:, t - CL:t], self._pad(ctx['roll_all'][:, t:end], ctx['roll'], n)], 1)
        fv = torch.cat([ctx['v_all'][:, t - CL:t], self._pad(ctx['v_all'][:, t:end], ctx['v'], n)], 1)
        fa = torch.cat([ctx['a_all'][:, t - CL:t], self._pad(ctx['a_all'][:, t:end], ctx['a'], n)], 1)
        act_h = torch.stack(ctx['act_win'], 1)
        lat_h = torch.stack(ctx['lat_win'], 1)
        cur0 = ctx['cur'].detach()
        u_prev = act_h[:, -1]

        u = torch.cat([self.plan[:, 1:], self.plan[:, -1:]], 1).detach()
        D = self.D
        eye = torch.eye(n, device=self.dev).expand(self.B, n, n)
        for _ in range(self.iters):
            uu = u.detach().clone().requires_grad_(True)
            lat0 = self._freerun(uu, act_h, lat_h, fr, fv, fa, cur0)
            # exact Jacobian: one backward per horizon output (banded, cheap for H~20)
            rows = []
            for h in range(n):
                go = torch.zeros_like(lat0); go[:, h] = 1.0
                (gr,) = torch.autograd.grad(lat0, uu, grad_outputs=go, retain_graph=(h < n - 1))
                rows.append(gr)
            J = torch.stack(rows, 1)                       # [B,H,H]
            lat0 = lat0.detach()
            e = lat0 - tgt                                  # tracking residual
            d0 = torch.zeros_like(lat0); d0[:, 0] = cur0
            f = (D @ lat0.unsqueeze(-1)).squeeze(-1) - d0   # jerk residual
            DJ = D.unsqueeze(0) @ J
            A = W_TRACK * (J.transpose(1, 2) @ J) + W_JERK * (DJ.transpose(1, 2) @ DJ)
            b = W_TRACK * (J.transpose(1, 2) @ e.unsqueeze(-1)).squeeze(-1) \
                + W_JERK * (DJ.transpose(1, 2) @ f.unsqueeze(-1)).squeeze(-1)
            if self.r_du:
                du_now = torch.cat([(u[:, :1] - u_prev[:, None]), u[:, 1:] - u[:, :-1]], 1)
                A = A + self.r_du * (D.T @ D).unsqueeze(0)
                b = b + self.r_du * (D.T.unsqueeze(0) @ du_now.unsqueeze(-1)).squeeze(-1)
            A = A + self.ridge * eye * A.diagonal(dim1=1, dim2=2).mean(1)[:, None, None].clamp(min=1.0)
            try:
                step = torch.linalg.solve(A, -b.unsqueeze(-1)).squeeze(-1)
            except Exception:
                break
            step = torch.nan_to_num(step, nan=0.0).clamp(-1.0, 1.0)     # trust region
            u = (u + self.damp * step).clamp(*STEER_RANGE)
        self.plan = u.detach()
        return self.plan[:, 0]


def evaluate(nseg=20, off=7000, H=20, iters=3, r_du=0.0, damp=1.0, bs=20,
             plant_is_surrogate=False, verbose=True, clamp='soft'):
    sur = Surrogate().to(DEV); sur.load_state_dict(torch.load('surrogate.pt')); sur.eval()
    for p in sur.parameters():
        p.requires_grad_(False)
    plant = None if plant_is_surrogate else Plant(device=DEV)
    tot = []
    files = ALL[off:off + nseg]
    for i in range(0, len(files), bs):
        segs = [load_segment(f) for f in files[i:i + bs]]
        B = len(segs)
        ctrl = GaussNewtonMPC(sur, B, DEV, H=H, iters=iters, r_du=r_du, damp=damp, clamp=clamp)
        if plant_is_surrogate:
            traj, target = _rollout_surrogate(sur, segs, ctrl, B)
        else:
            torch.manual_seed(0)
            traj, target = rollout(plant, segs, ctrl, mode='sample', stop=COST_END_IDX)
        l, j, c = cost(traj, target)
        tot.append(c)
        if verbose:
            print(f"  segs {i+B}/{len(files)}: lat={l.mean():.3f} jerk={j.mean():.2f} "
                  f"total={c.mean():.2f}", flush=True)
    return float(torch.cat(tot).mean())


def _rollout_surrogate(sur, segs, ctrl, B):
    """Closed loop where the SURROGATE is the plant (deterministic, perfectly modelled)."""
    T = COST_END_IDX
    def stk(k):
        return torch.tensor(np.stack([s[k][:T] for s in segs]), dtype=torch.float32, device=DEV)
    roll, v, a, target, steer0 = stk('roll'), stk('v'), stk('a'), stk('target'), stk('steer')
    lat_hist = [target[:, i] for i in range(CL)]
    act_hist = [steer0[:, i] for i in range(CL)]
    cur = lat_hist[-1]; traj = list(lat_hist)
    for t in range(CL, T):
        ctx = dict(target=target[:, t], cur=cur, roll=roll[:, t], v=v[:, t], a=a[:, t],
                   fut_lat=target[:, t + 1:min(t + 50, T)], fut_roll=roll[:, t + 1:min(t + 50, T)],
                   fut_v=v[:, t + 1:min(t + 50, T)], step=t,
                   act_win=act_hist[-CL:], lat_win=lat_hist[-CL:],
                   roll_all=roll, v_all=v, a_all=a, T=T)
        act = ctrl(ctx)
        if t < CONTROL_START_IDX:
            act = steer0[:, t]
        act = act.clamp(*STEER_RANGE); act_hist.append(act)
        with torch.no_grad():
            pred = sur(torch.stack(act_hist[-CL:], 1), roll[:, t - CL + 1:t + 1],
                       v[:, t - CL + 1:t + 1], a[:, t - CL + 1:t + 1],
                       torch.stack(lat_hist[-CL:], 1))
            pred = cur + (pred - cur).clamp(-MAX_ACC_DELTA, MAX_ACC_DELTA)
        cur = torch.where(torch.tensor(t >= CONTROL_START_IDX, device=DEV), pred, target[:, t])
        lat_hist.append(cur); traj.append(cur)
    return torch.stack(traj, 1), target


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--nseg', type=int, default=10)
    ap.add_argument('--H', type=int, default=20)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--rdu', type=float, default=0.0)
    ap.add_argument('--damp', type=float, default=1.0)
    ap.add_argument('--off', type=int, default=7000)
    ap.add_argument('--onsur', action='store_true', help='control the surrogate (perfect model)')
    ap.add_argument('--clamp', default='soft', choices=['hard','soft','none'])
    a = ap.parse_args()
    m = evaluate(a.nseg, a.off, a.H, a.iters, a.rdu, a.damp,
                 plant_is_surrogate=a.onsur, verbose=False, clamp=a.clamp)
    tag = 'SURROGATE(perfect model)' if a.onsur else 'REAL plant'
    print(f"GN-MPC on {tag}: H={a.H} iters={a.iters} rdu={a.rdu} clamp={a.clamp} "
          f"nseg={a.nseg} -> total={m:.2f}")
