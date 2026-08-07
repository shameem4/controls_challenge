"""Per-segment seed-aware plan optimiser: a DIFFICULTY MEASURE, not a controller.

The point. Every floor used so far has been analytic: the causal noise floor (31.24, or per segment
via `segfloor.py`) plus the Tikhonov trajectory optimum J*. Both assume the two costs ADD, which is
untested and demonstrably wrong in at least one direction -- a seed-aware plan can score BELOW the
causal floor, because knowing the noise draw lets you pre-compensate for it. So the honest per-segment
difficulty measure is: what does the best plan we can actually find cost on this segment? That
accounts for trajectory shape, the realised noise draw, and the plant's dynamics together, with no
additivity assumption.

This is a seed exploit by construction (`tinyphysics.py:116` seeds from md5(filename)) and is NOT
submittable. It is a measuring instrument.

Two defects in `steer_lookup.py` made it unusable for per-segment work; both are fixed here.

  PER-SEGMENT ACCEPTANCE.  steer_lookup accepts a sweep on the BATCH MEAN, so a sweep that helps the
    average is kept even on segments it hurt. Measured: on 128 segments the resulting "oracle" was
    WORSE than the warm start on a whole quartile (42.99 -> 50.13), which would have been read as
    "the controller already beats the oracle there". Here every segment keeps its own best plan and
    its own best cost.

  SWEEPS THAT ACTUALLY DIFFER.  steer_lookup's sweep is deterministic, and a rejected sweep restores
    the previous plan -- so sweeps 2..N replay sweep 1 exactly and change nothing (observed: 42.570
    four times in a row). Here each sweep anneals the step size and visits positions in a shuffled
    order, so extra sweeps do real work.

Usage: python steer_opt.py <nseg> [start] [tag]    env: K, HZ, SPAN, SWEEPS, WARM
"""
import sys, os, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, STEER_RANGE, DEL_T
from oracle_build import Stepper
from steer_lookup import score, window_cost, run_plan

DEV = 'cuda'
K = int(os.environ.get('K', 9))
HZ = int(os.environ.get('HZ', 25))
SPAN = float(os.environ.get('SPAN', 0.30))
SWEEPS = int(os.environ.get('SWEEPS', 8))
WARM = os.environ.get('WARM', 'cnn_v4.pt')


@torch.no_grad()
def sweep(plant, segs, plan, T, deltas, order_seed):
    """One coordinate pass. Positions are visited in a shuffled order so repeated sweeps differ.

    The rollout must still run forward in time, so 'shuffled order' means each position is optimised
    against the current plan but only a random SUBSET is updated per sweep; over sweeps every
    position is revisited under a different context. That keeps sweeps informative without breaking
    causality of the replay.
    """
    B = len(segs)
    rng = np.random.default_rng(order_seed)
    touch = rng.random(T) < 0.75          # 75% of positions per sweep, different each sweep
    torch.manual_seed(0)
    S = Stepper(plant, segs, T)
    while S.t < CONTROL_START_IDX:
        S.step(plan[:, S.t])
    while S.t < T:
        t = S.t
        if not touch[t]:
            S.step(plan[:, t]); continue
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
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    tag = sys.argv[3] if len(sys.argv) > 3 else 'so'
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV)
    sd = torch.load(WARM, map_location=DEV)
    ch = sd['ff_conv.0.weight'].shape[0]
    if ch != 32:
        net = AblNet('PM', ch=ch, fb_hidden=ch, res_hidden=ch).to(DEV)
    net.load_state_dict(sd); net.eval()
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    files = ALL[start:start + nseg]
    segs = [load_segment(f) for f in files]
    B, T = len(segs), COST_END_IDX
    print(f'[{tag}] segs={B} start={start} K={K} HZ={HZ} SPAN={SPAN} SWEEPS={SWEEPS} warm={WARM}', flush=True)

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
    warm = score(c, tg)
    best_cost = warm.clone()
    best_plan = plan.clone()
    print(f'[{tag}] warm start   mean {warm.mean():8.3f}  median {warm.median():7.3f}', flush=True)

    for sw in range(SWEEPS):
        span = SPAN * (0.6 ** sw)                       # anneal the step size
        deltas = torch.linspace(-span, span, K, device=DEV)
        plan = sweep(plant, segs, best_plan.clone(), T, deltas, order_seed=sw)
        c, tg, _, _ = run_plan(plant, segs, plan, T)
        cur = score(c, tg)
        imp = cur < best_cost - 1e-6                    # PER-SEGMENT acceptance
        best_plan[imp] = plan[imp]
        best_cost = torch.where(imp, cur, best_cost)
        print(f'[{tag}] sweep {sw + 1} span={span:.4f}  mean {best_cost.mean():8.3f}  '
              f'median {best_cost.median():7.3f}  improved {int(imp.sum())}/{B}', flush=True)

    c, tg, _, _ = run_plan(plant, segs, best_plan, T)
    fin = score(c, tg)
    assert torch.allclose(fin, best_cost, atol=1e-3), 'replay does not reproduce the selected plans'
    print(f'[{tag}] FINAL  mean {fin.mean():8.3f}  median {fin.median():7.3f}  '
          f'never-improved {int((best_cost >= warm - 1e-6).sum())}/{B}', flush=True)
    np.savez(f'{tag}_{start}_{nseg}.npz', cost=fin.cpu().numpy(), warm=warm.cpu().numpy(),
             plan=best_plan.cpu().numpy(), files=np.array([str(f) for f in files]))
    print(f'[{tag}] saved {tag}_{start}_{nseg}.npz', flush=True)


if __name__ == '__main__':
    main()
