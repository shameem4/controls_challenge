"""Per-segment optimiser by SEQUENTIAL LINEARISATION -- a sharper difficulty instrument.

Borrowed from `nurikserikbayev/comma-controls-challenge`, which reports 29.37 where our
coordinate-descent `steer_opt.py` converges at 38.25. Like `steer_opt` this is a seed exploit by
construction and NOT a submittable controller; it exists to measure how much cost is actually
recoverable per segment, which every targeting experiment so far has had to guess at.

WHY THIS SHOULD BEAT COORDINATE DESCENT. The benchmark cost is exactly quadratic in the lataccel
trajectory, and the trajectory is locally linear in the actions. So around any operating plan the
whole 400-step problem has a closed-form optimum -- one linear solve moves every action at once,
where coordinate descent moves them one at a time against a 25-step window and cannot see the
global trade.

    c ~= c0 + H (u - u0)                      H = lower-triangular Toeplitz of the impulse response
    J = A||c - tau||^2 + B||D c||^2           A = 5000/N,  B = 100/((N-1) DEL_T^2)
    (A H'H + B H'D'D H + lam I) d = -(A H'(c0 - tau) + B H'D'D c0)

`lam` is a Levenberg-Marquardt trust region: the linear model is only local, and the plant is
chaotic in the sense that a large action change lands on a different token sequence entirely.

H is shared across segments, so the 400x400 system is factorised ONCE per lam and every segment is a
batched triangular solve. That is why this is cheap despite looking heavier than coordinate descent.

CORRECTNESS. The linear model is used ONLY to propose a step. Every candidate is replayed through the
true plant with the RNG reset, and accepted per segment only if the true cost improves -- the same
discipline as `steer_opt`, and the reason a wrong H costs iterations rather than correctness. The
same two defects fixed there are avoided here by construction: acceptance is per segment (not on the
batch mean), and successive iterations differ because each re-linearises around a new trajectory.

Usage: python steer_seq.py <nseg> [start] [tag]   env: ITERS, WARM, HZ_H, LAMS
"""
import sys, os, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy
from tinyphysics import (CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX, STEER_RANGE,
                         DEL_T, MAX_ACC_DELTA)
from oracle_build import Stepper
from steer_lookup import score, run_plan

DEV = 'cuda'
ITERS = int(os.environ.get('ITERS', 12))
WARM = os.environ.get('WARM', 'cnn_v4.pt')
HZ_H = int(os.environ.get('HZ_H', 30))          # response length; must be long enough to settle
LAMS = [float(x) for x in os.environ.get('LAMS', '3.0,1.0,0.3,0.1').split(',')]
STEPS = [float(x) for x in os.environ.get('STEPS', '1.0,0.5,0.25').split(',')]


@torch.no_grad()
def impulse_response(plant, segs, net, delta=0.05, nsamp=24, seed=0):
    """Mean impulse response du -> dlataccel, measured in `expected` mode along real trajectories.

    Measured as a SUSTAINED offset and then differenced, not as a one-step impulse. A one-step
    impulse measured over 12 steps gave a kernel whose tail never decayed (flat at ~0.37, sum 4.25)
    against a verified steady-state DC gain of 2.4 -- the plant is autoregressive on its own lataccel
    output, so a one-step probe has not settled within a short window and the truncated kernel implies
    every action has a permanent effect. The Toeplitz model built from it was badly wrong and the
    optimiser stalled after 3 accepted steps. Differencing a sustained step response settles properly
    and reconciles with `verify_plant.py`.

    `expected` mode is essential: the plant emits a discrete token, so under sampling both arms
    usually draw the SAME token and the difference is exactly zero (this produced a median DC gain of
    0.000 in an earlier paired-perturbation attempt). The conditional mean has no such problem.
    """
    B, T = len(segs), COST_END_IDX
    torch.manual_seed(seed)
    S = Stepper(plant, segs, T)
    pol = AblPolicy(net, B, DEV)
    rng = np.random.default_rng(0)
    times = np.sort(rng.choice(np.arange(CONTROL_START_IDX + 5, T - HZ_H - 2), nsamp, replace=False))
    acc, n = torch.zeros(HZ_H, device=DEV), 0
    ptr = 0
    while S.t < T:
        t = S.t
        u = pol(S.ctx())
        if ptr < len(times) and times[ptr] == t:
            snap = S.snapshot()
            pstate = (pol.integ.clone(), pol.prev.clone(), [a.clone() for a in pol.pact])
            S.mode = 'expected'
            base, acts = [], []
            uu = u
            for j in range(HZ_H):
                acts.append(uu); base.append(S.step(uu)); uu = pol(S.ctx())
            S.restore(snap)
            pol.integ, pol.prev, pol.pact = pstate[0].clone(), pstate[1].clone(), [a.clone() for a in pstate[2]]
            pert = []
            for j in range(HZ_H):                      # SUSTAINED offset, differenced below
                pert.append(S.step((acts[j] + delta).clamp(*STEER_RANGE)))
            S.restore(snap)
            pol.integ, pol.prev, pol.pact = pstate[0].clone(), pstate[1].clone(), [a.clone() for a in pstate[2]]
            S.mode = 'sample'
            step_resp = ((torch.stack(pert, 1) - torch.stack(base, 1)) / delta).mean(0)
            acc += torch.cat([step_resp[:1], step_resp[1:] - step_resp[:-1]])   # impulse = d(step)
            n += 1
            ptr += 1
        S.step(u)
    return (acc / max(n, 1)).cpu().numpy()


def build_system(h, N, lams):
    """Pre-factorise (A H'H + B H'D'D H + lam I) once per lam; H is shared across segments."""
    H = torch.zeros(N, N, device=DEV, dtype=torch.float64)
    for k, hk in enumerate(h):
        if k >= N:
            break
        idx = torch.arange(N - k, device=DEV)
        H[idx + k, idx] = float(hk)
    D = torch.zeros(N - 1, N, device=DEV, dtype=torch.float64)
    D[torch.arange(N - 1), torch.arange(N - 1)] = -1.0
    D[torch.arange(N - 1), torch.arange(N - 1) + 1] = 1.0
    A = 5000.0 / N
    Bc = 100.0 / ((N - 1) * DEL_T ** 2)
    HtH = H.T @ H
    DtD = D.T @ D
    HtDtDH = H.T @ DtD @ H
    base = A * HtH + Bc * HtDtDH
    chols = {}
    for lam in lams:
        chols[lam] = torch.linalg.cholesky(base + lam * torch.eye(N, device=DEV, dtype=torch.float64))
    return H, DtD, A, Bc, chols


def main():
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    tag = sys.argv[3] if len(sys.argv) > 3 else 'sq'
    plant = Plant(device=DEV)
    sd = torch.load(WARM, map_location=DEV)
    ch = sd['ff_conv.0.weight'].shape[0]
    net = AblNet('PM', ch=ch, fb_hidden=ch, res_hidden=ch).to(DEV)
    net.load_state_dict(sd); net.eval()
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    files = ALL[start:start + nseg]
    segs = [load_segment(f) for f in files]
    B, T = len(segs), COST_END_IDX
    N = T - CONTROL_START_IDX
    print(f'[{tag}] segs={B} start={start} ITERS={ITERS} warm={WARM} N={N}', flush=True)

    h = impulse_response(plant, segs, net)
    print(f'[{tag}] impulse response: {np.round(h, 4)}   sum(DC gain) = {h.sum():.3f}', flush=True)
    H, DtD, A, Bc, chols = build_system(h, N, LAMS)

    with torch.no_grad():                       # warm start = the controller's own actions
        torch.manual_seed(0)
        S = Stepper(plant, segs, T)
        pol = AblPolicy(net, B, DEV)
        plan = torch.zeros(B, T, device=DEV)
        while S.t < T:
            plan[:, S.t] = pol(S.ctx()); S.step(plan[:, S.t])
    plan = plan.detach()

    c, tg, _, _ = run_plan(plant, segs, plan, T)
    best_cost, best_plan = score(c, tg).clone(), plan.clone()
    print(f'[{tag}] warm start   mean {best_cost.mean():8.3f}  median {best_cost.median():7.3f}', flush=True)

    for it in range(ITERS):
        c, tg, _, _ = run_plan(plant, segs, best_plan, T)
        c64 = c.double(); tg64 = tg.double()
        e0 = (c64 - tg64)
        rhs = -(A * (H.T @ e0.T) + Bc * (H.T @ (DtD @ c64.T)))       # [N,B]
        improved_any = 0
        for lam in LAMS:
            d = torch.cholesky_solve(rhs, chols[lam]).T.float()       # [B,N]
            for a in STEPS:
                cand = best_plan.clone()
                cand[:, CONTROL_START_IDX:T] = (cand[:, CONTROL_START_IDX:T] + a * d).clamp(*STEER_RANGE)
                cc, ctg, _, _ = run_plan(plant, segs, cand, T)
                sc = score(cc, ctg)
                imp = sc < best_cost - 1e-6                            # PER-SEGMENT acceptance
                if imp.any():
                    best_plan[imp] = cand[imp]
                    best_cost = torch.where(imp, sc, best_cost)
                    improved_any += int(imp.sum())
        print(f'[{tag}] iter {it + 1:2d}  mean {best_cost.mean():8.3f}  median {best_cost.median():7.3f}  '
              f'accepted {improved_any}', flush=True)

    c, tg, _, _ = run_plan(plant, segs, best_plan, T)
    fin = score(c, tg)
    assert torch.allclose(fin, best_cost, atol=1e-3), 'replay does not reproduce the selected plans'
    print(f'[{tag}] FINAL  mean {fin.mean():8.3f}  median {fin.median():7.3f}', flush=True)
    np.savez(f'{tag}_{start}_{nseg}.npz', cost=fin.cpu().numpy(), plan=best_plan.cpu().numpy(),
             h=h, files=np.array([str(f) for f in files]))
    print(f'[{tag}] saved {tag}_{start}_{nseg}.npz', flush=True)


if __name__ == '__main__':
    main()
