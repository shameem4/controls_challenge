"""Receding-horizon planner that optimises the next 20 actions for mean AND variance.

Three things distinguish this from the MPC attempts that failed here before.

**It plans in `expected` mode.** The earlier neural-plant MPC sampled, and sampling is what made the
landscape hostile: an action change flips which bin a fixed uniform draw selects and the trajectory
jumps (`FINDINGS_SEQLIN.md` measured +3.67 mean cost from perturbing actions by 1e-4). The conditional
mean is deterministic, smooth and differentiable, so planning on it has no such cliff.

**It scores the variance, which is part of the true objective.** The expected cost decomposes exactly:

    E[(c - tau)^2] = (mu - tau)^2 + sigma^2
    E[(c_k - c_{k-1})^2] = (mu_k - mu_{k-1})^2 + sigma_k^2 + sigma_{k-1}^2

Every controller in this project optimised only the first half of each. `FINDINGS_ENDOGENOUS_NOISE.md`
showed sigma^2 is genuinely controllable -- 0.02 of action jitter raises it 30% -- so the second half
is a real term with a real gradient, not a constant to be dropped.

**It plans a smooth continuation, not 20 independent scalars.** The action enters a 20-step context
window; jittering entries incoherently is what produces both chaos and excess plant noise. The plan is
parameterised by a few coefficients on a smooth basis, so every candidate is a coherent continuation
of the history.

The horizon is the plant's context length (20) on purpose: actions further out cannot influence the
current emission at all.

Usage: python plan_var.py [nseg] [start]   env: H, KSTEP, LR, WVAR, NBASIS, LAM_U
"""
import sys, os, numpy as np, torch
import torch.nn.functional as F
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy
from tinyphysics import (CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX, FUTURE_PLAN_STEPS,
                         STEER_RANGE, MAX_ACC_DELTA, DEL_T)

DEV = 'cuda'
TEMP = 0.8
H = int(os.environ.get('H', 20))            # planning horizon = plant context length
KSTEP = int(os.environ.get('KSTEP', 4))     # gradient steps per control step
LR = float(os.environ.get('LR', 0.05))
WVAR = float(os.environ.get('WVAR', 1.0))   # weight on the variance term (1.0 = the true objective)
NBASIS = int(os.environ.get('NBASIS', 3))   # smooth basis size: constant, ramp, quadratic
LAM_U = float(os.environ.get('LAM_U', 0.0))
WARMSTART = int(os.environ.get('WARMSTART', 1))
# BETA: plan-consistency filter. The emitted action blends the newly optimised plan with the PREVIOUS
# plan's prediction FOR THE SAME TIMESTEP, so unlike an EMA on the raw action it adds no lag -- both
# terms estimate the same time index. An EMA on cnn_v4's action was catastrophic for exactly that
# reason (tracking 28.63 -> 111.01 at alpha=0.85).
BETA = float(os.environ.get('BETA', 0.0))


def basis(H, n):
    """Smooth polynomial basis over the horizon, normalised so coefficients are comparable."""
    k = torch.arange(H, device=DEV, dtype=torch.float32) / max(H - 1, 1)
    return torch.stack([k ** j for j in range(n)], 0)          # [n, H]


@torch.no_grad()
def policy_base(plant, net, ctx_act, ctx_roll, ctx_v, ctx_a, ctx_lat, cur, target, roll, v, a, t, T):
    """Roll the POLICY forward H steps in expected mode to get a realistic base plan.

    Holding the current action constant over the horizon (the first version) makes the planner
    optimise its first action against a future that will never happen -- and it showed: more gradient
    steps made the closed loop worse (total 62 at 4 steps, 181 at 12). A base plan the policy would
    actually execute removes that mismatch.
    """
    B = ctx_act.shape[0]
    act, lat = ctx_act.clone(), ctx_lat.clone()
    pol = AblPolicy(net, B, DEV)
    cu = cur
    out = []
    for k in range(H):
        tt = min(t + k, T - 1)
        fe = min(tt + FUTURE_PLAN_STEPS, T)
        fl = target[:, tt + 1:fe]
        ctx = dict(target=target[:, tt], cur=cu, roll=roll[:, tt], v=v[:, tt], a=a[:, tt],
                   fut_lat=fl, fut_roll=roll[:, tt + 1:fe], fut_v=v[:, tt + 1:fe], step=tt)
        u = pol(ctx).clamp(*STEER_RANGE)
        out.append(u)
        act = torch.cat([act[:, 1:], u.unsqueeze(1)], 1)
        st = torch.stack([act, ctx_roll, ctx_v, ctx_a], -1)
        logits = plant.m(st, plant.tokenize(lat))[:, -1]
        p = F.softmax(logits / TEMP, -1)
        mu = (p * plant.bins).sum(-1)
        mu = torch.clamp(mu, cu - MAX_ACC_DELTA, cu + MAX_ACC_DELTA)
        lat = torch.cat([lat[:, 1:], mu.unsqueeze(1)], 1)
        cu = mu
    return torch.stack(out, 1)                                  # [B, H]


def plan_step(plant, ctx_act, ctx_roll, ctx_v, ctx_a, ctx_lat, cur, tau, Bmat, u_base, theta_prev=None):
    """Optimise smooth coefficients over the horizon; return the first action of the best plan.

    All arithmetic is on the conditional mean and variance, so this is deterministic and
    differentiable end to end.
    """
    Bn, Hh = ctx_act.shape[0], Bmat.shape[1]
    # WARM START across control steps. Resetting theta to zero every step means consecutive actions
    # come from different partially-converged optima, and that inconsistency IS action jitter -- which
    # FINDINGS_ENDOGENOUS_NOISE showed raises the plant's own variance ~30%. Standard MPC practice is
    # to carry the plan forward; omitting it made the planner inflate the noise it was minimising.
    if theta_prev is None or not WARMSTART:
        theta = torch.zeros(Bn, Bmat.shape[0], device=DEV, requires_grad=True)
    else:
        theta = theta_prev.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=LR)
    A = 5000.0 / Hh
    Bc = 100.0 / (max(Hh - 1, 1) * DEL_T ** 2)
    for _ in range(KSTEP):
        opt.zero_grad()
        du = theta @ Bmat                                   # [B, H] smooth correction
        u = (u_base + du).clamp(*STEER_RANGE)                # u_base is now [B, H]
        act = ctx_act.clone(); lat = ctx_lat.clone()
        prev_mu = cur
        loss = 0.0
        for k in range(Hh):
            act = torch.cat([act[:, 1:], u[:, k:k + 1]], 1)
            st = torch.stack([act, ctx_roll, ctx_v, ctx_a], -1)
            logits = plant.m(st, plant.tokenize(lat))[:, -1]
            p = F.softmax(logits / TEMP, -1)
            mu = (p * plant.bins).sum(-1)
            var = (p * plant.bins ** 2).sum(-1) - mu ** 2
            mu_c = torch.clamp(mu, prev_mu - MAX_ACC_DELTA, prev_mu + MAX_ACC_DELTA)
            # E[(c-tau)^2] = (mu-tau)^2 + var ;  E[(dc)^2] = (dmu)^2 + var_k + var_{k-1}
            loss = loss + A * ((mu_c - tau[:, k]) ** 2 + WVAR * var).sum()
            loss = loss + Bc * ((mu_c - prev_mu) ** 2 + WVAR * var).sum()
            lat = torch.cat([lat[:, 1:], mu_c.unsqueeze(1)], 1)
            prev_mu = mu_c
        if LAM_U:
            loss = loss + LAM_U * (du[:, 1:] - du[:, :-1]).pow(2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (u_base + theta @ Bmat).clamp(*STEER_RANGE), theta.detach()


@torch.no_grad()
def _ctx_tensors(act, sr, sv, sa, lat):
    return (torch.stack(act[-CONTEXT_LENGTH:], 1), torch.stack(sr[-CONTEXT_LENGTH:], 1),
            torch.stack(sv[-CONTEXT_LENGTH:], 1), torch.stack(sa[-CONTEXT_LENGTH:], 1),
            torch.stack(lat[-CONTEXT_LENGTH:], 1))


def run(plant, segs, net, planner=True, seed=0):
    B, T = len(segs), COST_END_IDX
    torch.manual_seed(seed)
    stk = lambda k: torch.tensor(np.stack([s[k][:T] for s in segs]), dtype=torch.float32, device=DEV)
    roll, v, a, target, steer0 = stk('roll'), stk('v'), stk('a'), stk('target'), stk('steer')
    lat = [target[:, i] for i in range(CONTEXT_LENGTH)]
    act = [steer0[:, i] for i in range(CONTEXT_LENGTH)]
    sr = [roll[:, i] for i in range(CONTEXT_LENGTH)]
    sv = [v[:, i] for i in range(CONTEXT_LENGTH)]
    sa = [a[:, i] for i in range(CONTEXT_LENGTH)]
    cur = lat[-1]
    pol = AblPolicy(net, B, DEV)
    Bmat = basis(H, NBASIS)
    traj, vars_, dus = [], [], []
    prev_u = act[-1]
    theta_prev = None
    plan_prev = None
    for t in range(CONTEXT_LENGTH, T):
        fe = min(t + FUTURE_PLAN_STEPS, T)
        ctx = dict(target=target[:, t], cur=cur, roll=roll[:, t], v=v[:, t], a=a[:, t],
                   fut_lat=target[:, t + 1:fe], fut_roll=roll[:, t + 1:fe],
                   fut_v=v[:, t + 1:fe], step=t)
        with torch.no_grad():
            u = pol(ctx)
        if planner and t >= CONTROL_START_IDX:
            ca, cr, cv, cak, cl = _ctx_tensors(act, sr, sv, sa, lat)
            hi = min(t + H, T)
            tau = target[:, t:hi]
            if tau.shape[1] < H:                       # pad the tail with the last target
                tau = torch.cat([tau, tau[:, -1:].expand(B, H - tau.shape[1])], 1)
            ub = policy_base(plant, net, ca, cr, cv, cak, cl, cur, target, roll, v, a, t, T)
            plan_new, theta_prev = plan_step(plant, ca, cr, cv, cak, cl, cur, tau, Bmat, ub, theta_prev)
            if BETA > 0 and plan_prev is not None:
                # shift the previous plan so entry k refers to the same absolute time as plan_new[k]
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
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV)
    net.load_state_dict(torch.load('cnn_v4.pt', map_location=DEV)); net.eval()
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    segs = [load_segment(f) for f in ALL[start:start + nseg]]
    print(f'  segs={nseg} H={H} KSTEP={KSTEP} LR={LR} WVAR={WVAR} NBASIS={NBASIS} '
          f'WARMSTART={WARMSTART} BETA={BETA}', flush=True)
    print(f'  {"arm":28} {"track":>8} {"jerk":>8} {"total":>8} {"E[Var]":>10} {"mean|du|":>9}', flush=True)
    l, j, vv, du = run(plant, segs, net, planner=False)
    print(f'  {"cnn_v4 (baseline)":28} {l.mean():8.2f} {j.mean():8.2f} {(l + j).mean():8.2f} '
          f'{vv:10.6f} {du:9.5f}', flush=True)
    l, j, vv, du = run(plant, segs, net, planner=True)
    print(f'  {"planner":28} {l.mean():8.2f} {j.mean():8.2f} {(l + j).mean():8.2f} '
          f'{vv:10.6f} {du:9.5f}', flush=True)


if __name__ == '__main__':
    main()
