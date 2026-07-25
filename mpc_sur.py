"""Receding-horizon MPC that plans through the smooth differentiable SURROGATE.

Why this is expected to work where MPC on the raw plant failed: the raw plant is chaotic to plan
against (stochastic token sampling + a discrete embedding bottleneck), so candidate action
sequences cannot be ranked. The surrogate is a deterministic smooth MLP -- neither pathology --
and it predicts the plant's conditional mean, which is the RMS-optimal predictor. Measured
multi-step accuracy: 1.33x the plant's own predictive floor at H=30, vs 2.20x for the LPV.

Each control step: warm-start the action sequence from the previous plan, run K Adam steps on the
exact challenge cost back-propagated through an H-step free-run of the surrogate, apply u[0].
Everything is batched across segments on the GPU, which is what makes this tractable.
"""
import sys, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment, rollout, cost
from nets import AblNet, AblPolicy
from surrogate import Surrogate
from tinyphysics import (CONTEXT_LENGTH as CL, CONTROL_START_IDX, COST_END_IDX,
                         MAX_ACC_DELTA, STEER_RANGE)

DEV = 'cuda'
ALL = sorted(Path('data/SYNTHETIC').iterdir())
W_TRACK, W_JERK = 5000.0, 10000.0        # exact challenge weights


class SurrogateMPC:
    """Batched gradient MPC. Usable as a controller inside torch_sim.rollout."""

    def __init__(self, sur, B, dev, H=20, iters=6, lr=0.15, r_du=0.0, warm=None):
        self.sur, self.B, self.dev = sur, B, dev
        self.H, self.iters, self.lr, self.r_du = H, iters, lr, r_du
        self.plan = torch.zeros(B, H, device=dev)
        self.warm = warm                  # optional policy to initialise the plan
        self.wpol = None

    def _pad(self, arr, head, n):
        """arr [B,k] -> [B,n], edge-padded (or filled with head if empty)."""
        k = arr.shape[1]
        if k >= n:
            return arr[:, :n]
        if k == 0:
            return head[:, None].expand(self.B, n)
        return torch.cat([arr, arr[:, -1:].expand(self.B, n - k)], 1)

    def __call__(self, ctx):
        t, T, n = ctx['step'], ctx['T'], self.H
        if t < CONTROL_START_IDX:                 # warmup actions are overridden by the sim anyway
            return torch.zeros(self.B, device=self.dev)

        # exogenous inputs over the horizon (known: current + future plan)
        tgt = torch.cat([ctx['target'][:, None], self._pad(ctx['fut_lat'], ctx['target'], n)], 1)[:, :n]
        end = min(t + n, T)
        roll_f = ctx['roll_all'][:, t:end]; v_f = ctx['v_all'][:, t:end]; a_f = ctx['a_all'][:, t:end]
        roll_f = self._pad(roll_f, ctx['roll'], n)
        v_f = self._pad(v_f, ctx['v'], n)
        a_f = self._pad(a_f, ctx['a'], n)
        # history windows (applied actions / lataccels), needed by the surrogate's context
        act_h = torch.stack(ctx['act_win'], 1)     # [B,CL]  actions t-CL..t-1
        lat_h = torch.stack(ctx['lat_win'], 1)     # [B,CL]  lataccels t-CL..t-1
        roll_h = ctx['roll_all'][:, t - CL:t]; v_h = ctx['v_all'][:, t - CL:t]; a_h = ctx['a_all'][:, t - CL:t]

        # roll/v/a are exogenous and fully known, so build their [B, CL+H] strips once and slice;
        # only the action strip (being optimised) and the lataccel history (fed back) recurse.
        full_roll = torch.cat([roll_h, roll_f], 1)
        full_v = torch.cat([v_h, v_f], 1)
        full_a = torch.cat([a_h, a_f], 1)

        # warm start: shift previous plan
        u = torch.cat([self.plan[:, 1:], self.plan[:, -1:]], 1).detach().clone()
        u.requires_grad_(True)
        opt = torch.optim.Adam([u], self.lr)
        cur0 = ctx['cur'].detach()
        for _ in range(self.iters):
            full_act = torch.cat([act_h, u], 1)          # [B, CL+H]
            lat_win = lat_h
            prev = cur0
            J = torch.zeros(self.B, device=self.dev)
            for h in range(n):
                s = slice(h + 1, h + 1 + CL)             # window ending at horizon step h
                pred = self.sur(full_act[:, s], full_roll[:, s], full_v[:, s], full_a[:, s], lat_win)
                pred = prev + (pred - prev).clamp(-MAX_ACC_DELTA, MAX_ACC_DELTA)   # plant slew clamp
                J = J + W_TRACK * (pred - tgt[:, h]) ** 2 + W_JERK * (pred - prev) ** 2
                lat_win = torch.cat([lat_win[:, 1:], pred[:, None]], 1)
                prev = pred
            if self.r_du:
                du = torch.cat([(u[:, :1] - act_h[:, -1:]), u[:, 1:] - u[:, :-1]], 1)
                J = J + self.r_du * du.pow(2).sum(1)
            loss = J.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([u], 50.0)
            opt.step()
            with torch.no_grad():
                u.data = torch.nan_to_num(u.data, nan=0.0).clamp(*STEER_RANGE)
        self.plan = u.detach()
        return self.plan[:, 0]


def evaluate(nseg=20, off=7000, H=20, iters=6, lr=0.15, r_du=0.0, bs=20, seed=0, quiet=False):
    plant = Plant(device=DEV)
    sur = Surrogate().to(DEV); sur.load_state_dict(torch.load('surrogate.pt')); sur.eval()
    for p in sur.parameters():
        p.requires_grad_(False)
    tot = []
    files = ALL[off:off + nseg]
    for i in range(0, len(files), bs):
        segs = [load_segment(f) for f in files[i:i + bs]]
        B = len(segs)
        torch.manual_seed(seed)
        ctrl = SurrogateMPC(sur, B, DEV, H=H, iters=iters, lr=lr, r_du=r_du)
        traj, target = rollout(plant, segs, ctrl, mode='sample', stop=COST_END_IDX)
        lat, jerk, c = cost(traj, target)
        tot.append(c)
        if not quiet:
            print(f"  segs {i+B}/{len(files)}: lat={lat.mean():.3f} jerk={jerk.mean():.2f} "
                  f"total={c.mean():.2f}", flush=True)
    c = torch.cat(tot)
    return float(c.mean())


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--nseg', type=int, default=20)
    ap.add_argument('--H', type=int, default=20)
    ap.add_argument('--iters', type=int, default=6)
    ap.add_argument('--lr', type=float, default=0.15)
    ap.add_argument('--rdu', type=float, default=0.0)
    ap.add_argument('--off', type=int, default=7000)
    a = ap.parse_args()
    m = evaluate(a.nseg, a.off, a.H, a.iters, a.lr, a.rdu)
    print(f"\nSurrogate-MPC  H={a.H} iters={a.iters} lr={a.lr} rdu={a.rdu} "
          f"nseg={a.nseg}: total={m:.2f}")
