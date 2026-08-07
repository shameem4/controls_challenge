"""Receding-horizon planner on SAMPLED rollouts (MPPI), against the expected-mode planner.

`FINDINGS_PLAN_VAR.md` isolated the expected-mode planner's failure to one cause: tracking. Every
smoothness measure was fixed (jerk, plant variance, action roughness all at or better than the trained
policy) and only aiming stayed broken -- the signature of certainty equivalence, where the plan is
optimised against a mean trajectory the system will not follow.

The principled fix is to optimise the TRUE expected cost instead of the cost of the mean:

    certainty equivalence   J(mean trajectory)
    what we want            E[ J(trajectory) ]           estimated by Monte Carlo

These differ whenever the cost is not linear in the trajectory, which here it is not -- it is
quadratic, so `E[J] = J(mean) + curvature * variance` and the mean-plan is systematically wrong.

WHY THE CHAOS RESULT DOES NOT FORBID THIS. `FINDINGS_SEQLIN.md` measured that perturbing actions by
1e-4 costs +3.67 -- but that is against ONE FIXED noise realisation, where a token flip is a
discontinuity. `E[cost | plan]` averages over realisations and is smooth in the plan. The problem
becomes estimator variance, which is what MPPI and common random numbers are for, rather than a
hostile landscape.

DESIGN
  * search space is the same 3 smooth basis coefficients, so it is 3-dimensional -- small enough for
    a sampling optimiser to cover with a handful of candidates
  * K candidates per step, M sampled rollouts each, scored on the true cost form
  * COMMON RANDOM NUMBERS across candidates: the same draws are used for every candidate at a given
    step, so candidate comparisons are paired and most of the estimator variance cancels
  * MPPI exponential-weighted update, warm-started across steps
  * the same plan-consistency filter (BETA) that was worth 29% in the expected-mode planner

Usage: python plan_mc.py [nseg] [start]   env: H, K, M, SIGMA, TEMP_MPPI, ITERS, BETA, MODE
"""
import sys, os, numpy as np, torch
import torch.nn.functional as F
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy
from tinyphysics import (CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX, FUTURE_PLAN_STEPS,
                         STEER_RANGE, MAX_ACC_DELTA, DEL_T)
from plan_var import policy_base

DEV = 'cuda'
TEMP = 0.8
H = int(os.environ.get('H', 20))
K = int(os.environ.get('K', 12))              # candidate plans per step
M = int(os.environ.get('M', 4))               # sampled rollouts per candidate
SIGMA = float(os.environ.get('SIGMA', 0.05))  # exploration std on the basis coefficients
TMPPI = float(os.environ.get('TMPPI', 20.0))  # MPPI temperature
ITERS = int(os.environ.get('ITERS', 2))       # MPPI refinement iterations per control step
NBASIS = int(os.environ.get('NBASIS', 3))
BETA = float(os.environ.get('BETA', 0.8))
MODE = os.environ.get('MODE', 'sample')       # 'sample' = Monte Carlo, 'expected' = the old planner
BASIS = os.environ.get('BASIS', 'poly')       # 'poly' = polynomial, 'block' = piecewise-constant


def make_basis(H, n, kind):
    """Plan parameterisation over the horizon.

    'block' is piecewise-constant over n equal segments. It scales gracefully from n=1 (a single
    number: hold one correction over the whole horizon) to n=H (fully independent per-step actions),
    which is exactly the resolution question. A polynomial basis cannot answer it -- past degree ~4
    the k**j columns are numerically collinear.
    """
    if kind == 'poly':
        k = torch.arange(H, device=DEV, dtype=torch.float32) / max(H - 1, 1)
        return torch.stack([k ** j for j in range(n)], 0)
    B = torch.zeros(n, H, device=DEV)
    edges = torch.linspace(0, H, n + 1).round().long()
    for i in range(n):
        B[i, edges[i]:max(edges[i + 1], edges[i] + 1)] = 1.0
    return B


@torch.no_grad()
def mppi_step(plant, ca, cr, cv, cak, cl, cur, tau, Bmat, u_base, theta, gen):
    """MPPI over the basis coefficients, scoring candidates on sampled rollouts."""
    B, Hh = ca.shape[0], Bmat.shape[1]
    nb = Bmat.shape[0]
    for _ in range(ITERS):
        # candidate 0 is the incumbent, so the update can never be worse than staying put
        dth = torch.randn(B, K, nb, device=DEV, generator=gen) * SIGMA
        dth[:, 0] = 0.0
        th = theta.unsqueeze(1) + dth                              # [B,K,nb]
        u = (u_base.unsqueeze(1) + th @ Bmat).clamp(*STEER_RANGE)  # [B,K,H]

        # replicate context over candidates and MC samples: [B,K,M] flattened
        rep = lambda x: x.unsqueeze(1).unsqueeze(1).expand(B, K, M, *x.shape[1:]).reshape(B * K * M, *x.shape[1:])
        act, lat = rep(ca), rep(cl)
        rr, vvv, aaa = rep(cr), rep(cv), rep(cak)
        prev = rep(cur.unsqueeze(-1)).squeeze(-1)
        uu = u.unsqueeze(2).expand(B, K, M, Hh).reshape(B * K * M, Hh)
        tt = tau.unsqueeze(1).unsqueeze(1).expand(B, K, M, Hh).reshape(B * K * M, Hh)
        cost = torch.zeros(B * K * M, device=DEV)
        A = 5000.0 / Hh
        Bc = 100.0 / (max(Hh - 1, 1) * DEL_T ** 2)
        for k in range(Hh):
            act = torch.cat([act[:, 1:], uu[:, k:k + 1]], 1)
            st = torch.stack([act, rr, vvv, aaa], -1)
            logits = plant.m(st, plant.tokenize(lat))[:, -1]
            p = F.softmax(logits / TEMP, -1)
            if MODE == 'expected':
                nxt = (p * plant.bins).sum(-1)
            else:
                # COMMON RANDOM NUMBERS: one uniform per (segment, MC sample, step), shared by every
                # candidate, so candidate comparisons are paired.
                un = torch.rand(B, 1, M, device=DEV, generator=gen).expand(B, K, M).reshape(-1)
                cdf = p.cumsum(-1)
                idx = torch.searchsorted(cdf, un.unsqueeze(-1)).squeeze(-1).clamp(0, p.shape[-1] - 1)
                nxt = plant.bins[idx]
            nxt = torch.clamp(nxt, prev - MAX_ACC_DELTA, prev + MAX_ACC_DELTA)
            cost = cost + A * (nxt - tt[:, k]) ** 2 + Bc * (nxt - prev) ** 2
            lat = torch.cat([lat[:, 1:], nxt.unsqueeze(1)], 1)
            prev = nxt
        J = cost.view(B, K, M).mean(2)                             # [B,K] MC estimate
        w = torch.softmax(-(J - J.min(1, keepdim=True).values) / TMPPI, dim=1)
        theta = (w.unsqueeze(-1) * th).sum(1)
    return theta


def run(plant, segs, net, planner=True, seed=0):
    B, T = len(segs), COST_END_IDX
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV); gen.manual_seed(seed + 1)
    stk = lambda k: torch.tensor(np.stack([s[k][:T] for s in segs]), dtype=torch.float32, device=DEV)
    roll, v, a, target, steer0 = stk('roll'), stk('v'), stk('a'), stk('target'), stk('steer')
    lat = [target[:, i] for i in range(CONTEXT_LENGTH)]
    act = [steer0[:, i] for i in range(CONTEXT_LENGTH)]
    sr = [roll[:, i] for i in range(CONTEXT_LENGTH)]
    sv = [v[:, i] for i in range(CONTEXT_LENGTH)]
    sa = [a[:, i] for i in range(CONTEXT_LENGTH)]
    cur = lat[-1]
    pol = AblPolicy(net, B, DEV)
    Bmat = make_basis(H, NBASIS, BASIS)
    theta = torch.zeros(B, NBASIS, device=DEV)
    plan_prev = None
    traj, vars_, dus = [], [], []
    prev_u = act[-1]
    for t in range(CONTEXT_LENGTH, T):
        fe = min(t + FUTURE_PLAN_STEPS, T)
        ctx = dict(target=target[:, t], cur=cur, roll=roll[:, t], v=v[:, t], a=a[:, t],
                   fut_lat=target[:, t + 1:fe], fut_roll=roll[:, t + 1:fe],
                   fut_v=v[:, t + 1:fe], step=t)
        with torch.no_grad():
            u = pol(ctx)
        if planner and t >= CONTROL_START_IDX:
            ca = torch.stack(act[-CONTEXT_LENGTH:], 1); cr = torch.stack(sr[-CONTEXT_LENGTH:], 1)
            cvv = torch.stack(sv[-CONTEXT_LENGTH:], 1); cak = torch.stack(sa[-CONTEXT_LENGTH:], 1)
            cl = torch.stack(lat[-CONTEXT_LENGTH:], 1)
            hi = min(t + H, T)
            tau = target[:, t:hi]
            if tau.shape[1] < H:
                tau = torch.cat([tau, tau[:, -1:].expand(B, H - tau.shape[1])], 1)
            ub = policy_base(plant, net, ca, cr, cvv, cak, cl, cur, target, roll, v, a, t, T)
            theta = mppi_step(plant, ca, cr, cvv, cak, cl, cur, tau, Bmat, ub, theta, gen)
            plan_new = (ub + theta @ Bmat).clamp(*STEER_RANGE)
            if BETA > 0 and plan_prev is not None:
                shifted = torch.cat([plan_prev[:, 1:], plan_prev[:, -1:]], 1)
                plan_new = BETA * shifted + (1.0 - BETA) * plan_new
            plan_prev = plan_new
            u = plan_new[:, 0]
        if t < CONTROL_START_IDX:
            u = steer0[:, t]
        u = u.clamp(*STEER_RANGE).detach()
        if t >= CONTROL_START_IDX:
            dus.append((u - prev_u).abs())
        prev_u = u
        act.append(u); sr.append(roll[:, t]); sv.append(v[:, t]); sa.append(a[:, t])
        with torch.no_grad():
            st = torch.stack([torch.stack(act[-CONTEXT_LENGTH:], 1), torch.stack(sr[-CONTEXT_LENGTH:], 1),
                              torch.stack(sv[-CONTEXT_LENGTH:], 1), torch.stack(sa[-CONTEXT_LENGTH:], 1)], -1)
            logits = plant.m(st, plant.tokenize(torch.stack(lat[-CONTEXT_LENGTH:], 1)))[:, -1]
            p = F.softmax(logits / TEMP, -1)
            mu = (p * plant.bins).sum(-1)
            var = (p * plant.bins ** 2).sum(-1) - mu ** 2
            idx = torch.multinomial(p, 1).squeeze(-1)
            pred = torch.clamp(plant.bins[idx], cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA)
        cur = pred if t >= CONTROL_START_IDX else target[:, t]
        lat.append(cur)
        if t >= CONTROL_START_IDX:
            traj.append(cur); vars_.append(var)
    c = torch.stack(traj, 1); tg = target[:, CONTROL_START_IDX:T]
    lat_c = ((c - tg) ** 2).mean(1) * 5000.0
    jerk = (((c[:, 1:] - c[:, :-1]) / DEL_T) ** 2).mean(1) * 100.0
    return lat_c, jerk, float(torch.cat(vars_).mean()), float(torch.stack(dus, 1).mean())


def main():
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV)
    net.load_state_dict(torch.load('cnn_v4.pt', map_location=DEV)); net.eval()
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    segs = [load_segment(f) for f in ALL[start:start + nseg]]
    print(f'  segs={nseg} H={H} K={K} M={M} SIGMA={SIGMA} TMPPI={TMPPI} ITERS={ITERS} '
          f'BETA={BETA} MODE={MODE} BASIS={BASIS} NBASIS={NBASIS}', flush=True)
    print(f'  {"arm":26} {"track":>8} {"jerk":>8} {"total":>8} {"E[Var]":>10} {"mean|du|":>9}', flush=True)
    l, j, vv, du = run(plant, segs, net, planner=False)
    print(f'  {"cnn_v4 (baseline)":26} {l.mean():8.2f} {j.mean():8.2f} {(l + j).mean():8.2f} '
          f'{vv:10.6f} {du:9.5f}', flush=True)
    l, j, vv, du = run(plant, segs, net, planner=True)
    print(f'  {"MPPI planner (" + MODE + ")":26} {l.mean():8.2f} {j.mean():8.2f} {(l + j).mean():8.2f} '
          f'{vv:10.6f} {du:9.5f}', flush=True)


if __name__ == '__main__':
    main()
