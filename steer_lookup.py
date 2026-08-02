"""Per-segment coordinate descent on steering commands -- the `steer_lookup` construction.

For each segment, optimise the ACTION SEQUENCE against that segment's fixed noise realisation, then
record (observation, optimal action) pairs as a behaviour-cloning teacher. The teacher is a seed
exploit by construction (`tinyphysics.py:116` seeds the RNG from md5(filename)); replaying it would
be a lookup table. We use it ONLY as a teacher -- the student reads observations alone and is causal.

Measured: warm start (cnn_v2) 43.624 -> 37.819 after one sweep on 16 segments. That is the first
construction in this project to beat cnn_v2, and it agrees with RyanL2/commacontrol's 39.9 for the
same method.

Four things had to be right; each silently produced garbage when it was not:

  RNG reset per evaluation   Otherwise a plan tuned against one draw sequence is scored against
                             another -- the warm start read 422 instead of 43.624.
  jerk across the boundary   The window cost must include the lataccel BEFORE the window, or
                             consecutive positions optimise independently and chatter becomes jerk
                             (an earlier build: 90.567 with jerk 46.7).
  future actions from the    An earlier greedy build let a base controller drive after the candidate,
  PLAN, not a controller     so each position was judged against a continuation that never happened.
  monotone acceptance        The HZ-step window is not the true objective, so an unguarded sweep can
                             make things worse: 37.819 -> 43.403 -> 674.732. A sweep is accepted only
                             if the FULL replayed cost improves.

Scoring and dataset capture share one code path (`run_plan`), and the capture pass asserts it
reproduces the selected plan's cost. A previous version had separate paths and they disagreed
(41.347 vs 37.819) -- which would have saved a dataset that did not correspond to the reported
teacher.

Evaluation is truncated at HZ steps deliberately: changing u[t] alters the sampled trajectory forever
after, but chaotically, so the far tail is noise rather than signal.

Usage: python steer_lookup.py <nseg> [start] [tag]     env: K, HZ, SPAN, SWEEPS
"""
import sys, os, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy, build_ff_window, build_multihorizon
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, STEER_RANGE, DEL_T
from oracle_build import Stepper

DEV = 'cuda'
K = int(os.environ.get('K', 9))
HZ = int(os.environ.get('HZ', 25))
SPAN = float(os.environ.get('SPAN', 0.25))
SWEEPS = int(os.environ.get('SWEEPS', 4))


def score(c, tg):
    lat = ((c - tg) ** 2).mean(1) * 5000.0
    jerk = (((c[:, 1:] - c[:, :-1]) / DEL_T) ** 2).mean(1) * 100.0
    return lat + jerk


def window_cost(lats, tgts, prev_lat):
    c = torch.stack(lats, 1); tg = torch.stack(tgts, 1)
    lat = ((c - tg) ** 2).mean(1) * 5000.0
    cj = torch.cat([prev_lat.unsqueeze(1), c], 1)
    jerk = (((cj[:, 1:] - cj[:, :-1]) / DEL_T) ** 2).mean(1) * 100.0
    return lat + jerk


@torch.no_grad()
def run_plan(plant, segs, plan, T, net=None, seed=0):
    """Replay `plan`; optionally capture the observations a causal controller would have seen.

    One code path for scoring and for dataset capture, so the saved dataset provably corresponds to
    the reported cost. `net` only supplies the reference controller whose integrator state is part of
    the observation vector -- it never drives the plant here.
    """
    torch.manual_seed(seed)
    S = Stepper(plant, segs, T)
    pol = AblPolicy(net, len(segs), DEV) if net is not None else None
    traj, tgts, obs, ucnn = [], [], [], []
    while S.t < T:
        t = S.t
        ctx = S.ctx()
        u_cnn = pol(ctx) if pol is not None else None
        if t >= CONTROL_START_IDX:
            if pol is not None:
                e = ctx['target'] - ctx['cur']
                ffw = build_ff_window(ctx['target'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
                mh = build_multihorizon(ctx['cur'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
                obs.append(torch.cat([torch.stack([e, pol.integ, ctx['v'], ctx['roll'], ctx['cur']], -1),
                                      ffw, mh], -1).cpu())
                ucnn.append(u_cnn.cpu())
            tgts.append(ctx['target'])
        S.step(plan[:, t])
        if t >= CONTROL_START_IDX:
            traj.append(S.cur)
    return (torch.stack(traj, 1), torch.stack(tgts, 1),
            torch.stack(obs, 1) if obs else None,
            torch.stack(ucnn, 1) if ucnn else None)


@torch.no_grad()
def sweep_once(plant, segs, plan, T, deltas):
    """One coordinate-descent pass: position t is re-optimised while t+1.. hold current plan values."""
    B = len(segs)
    torch.manual_seed(0)
    S = Stepper(plant, segs, T)
    while S.t < CONTROL_START_IDX:
        S.step(plan[:, S.t])
    while S.t < T:
        t = S.t
        snap = S.snapshot()
        best_c = torch.full((B,), float('inf'), device=DEV)
        best_u = plan[:, t].clone()
        for d in deltas:
            S.restore(snap)
            prev_lat = S.cur.clone()
            cand = (plan[:, t] + d).clamp(*STEER_RANGE)
            lats, tgts = [], []
            for j in range(min(HZ, T - t)):
                tgts.append(S.target[:, S.t])
                lats.append(S.step(cand if j == 0 else plan[:, t + j]))
            cc = window_cost(lats, tgts, prev_lat)
            better = cc < best_c
            best_c = torch.where(better, cc, best_c)
            best_u = torch.where(better, cand, best_u)
        plan[:, t] = best_u
        S.restore(snap)
        S.step(best_u)
    return plan


def main():
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    tag = sys.argv[3] if len(sys.argv) > 3 else 'sl'
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV); net.load_state_dict(torch.load('cnn_v2.pt', map_location=DEV)); net.eval()
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    files = ALL[start:start + nseg]
    segs = [load_segment(f) for f in files]
    B, T = len(segs), COST_END_IDX
    deltas = torch.linspace(-SPAN, SPAN, K, device=DEV)
    print(f'[{tag}] segs={B} start={start} K={K} HZ={HZ} SPAN={SPAN} SWEEPS={SWEEPS}', flush=True)

    # warm start from cnn_v2's own actions -- never from zero, which produced false optima twice here
    with torch.no_grad():
        torch.manual_seed(0)
        S = Stepper(plant, segs, T)
        pol = AblPolicy(net, B, DEV)
        plan = torch.zeros(B, T, device=DEV)
        while S.t < T:
            plan[:, S.t] = pol(S.ctx())
            S.step(plan[:, S.t])
    plan = plan.detach()

    c, tg, _, _ = run_plan(plant, segs, plan, T)
    best_cost, best_plan = score(c, tg).mean().item(), plan.clone()
    print(f'[{tag}] warm start (cnn_v2)   {best_cost:8.3f}', flush=True)

    for sw in range(SWEEPS):
        plan = sweep_once(plant, segs, plan.clone(), T, deltas)
        c, tg, _, _ = run_plan(plant, segs, plan, T)
        cur = score(c, tg).mean().item()
        if cur < best_cost - 1e-6:
            best_cost, best_plan = cur, plan.clone()
            print(f'[{tag}] sweep {sw + 1}             {cur:8.3f}  *', flush=True)
        else:
            print(f'[{tag}] sweep {sw + 1}             {cur:8.3f}   rejected (best {best_cost:.3f})', flush=True)
            plan = best_plan.clone()

    c, tg, obs, ucnn = run_plan(plant, segs, best_plan, T, net=net)
    fin = score(c, tg)
    print(f'[{tag}] FINAL teacher         {fin.mean().item():8.3f}  median {fin.median().item():.3f}', flush=True)
    assert abs(fin.mean().item() - best_cost) < 1e-3, \
        f'capture pass {fin.mean().item():.3f} != selected plan {best_cost:.3f}'
    ui = best_plan[:, CONTROL_START_IDX:T].cpu()
    np.savez(f'{tag}_{start}_{nseg}.npz', obs=obs.numpy(), u_ideal=ui.numpy(),
             u_cnn=ucnn.numpy(), traj=c.cpu().numpy(), tgt=tg.cpu().numpy(),
             cost=fin.cpu().numpy(), files=np.array([str(f) for f in files]))
    print(f'[{tag}] saved obs {tuple(obs.shape)}  u_ideal {tuple(ui.shape)}', flush=True)


if __name__ == '__main__':
    main()
