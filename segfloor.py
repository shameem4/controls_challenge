"""Per-segment causal cost floor, replacing the population-average constant 31.24.

WHY. `FINDINGS_FLOOR.md` derives the causal floor from the plant's noise variance:

    lataccel_floor = 5000 * E[sigma^2] * 3.3228      jerk_floor = 10000 * E[sigma^2]
    total_floor    = E[sigma^2] * 26614  = 31.24     at the population E[sigma^2] = 0.001174

That constant is an average over segments. Applied per segment it is wrong in both directions, and
the error is large: bucketing `cnn_v4` by cost/(31.24 + J*) put 814/2000 segments "below the floor"
(they drew quieter noise) and 396 in a "50-100% above" band. Those 396 are above +50% for EVERY
controller tested -- cnn_v3 94%, cnn_v2 94%, ff_pi_tau 92%, pid_boot 99% -- and per-segment excess
correlates 0.85-0.99 across architectures that share nothing. The band was sorting segments by noise
luck, not by controller deficiency, so "recoverable cost" computed against it is mostly not
recoverable.

WHAT THIS MEASURES. The plant emits a distribution over 1024 lataccel bins each step. Its conditional
variance is exact and needs no sampling:

    sigma^2_t = sum_i p_i * b_i^2  -  (sum_i p_i * b_i)^2 ,   p = softmax(logits / 0.8)

Averaged over the scored window that gives the segment's own sigma^2, hence its own floor.

CAVEAT. sigma^2 depends on the state, which depends on the controller, so this is a floor *along the
trajectory the controller actually drove*. It is therefore not perfectly controller-independent; the
script measures it under two structurally different controllers so the stability of the estimate can
be checked rather than assumed.

Usage: python segfloor.py <start> <n> [out.csv]
"""
import sys, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy
from tinyphysics import (CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX,
                         FUTURE_PLAN_STEPS, STEER_RANGE, MAX_ACC_DELTA)

DEV = 'cuda'
TEMP = 0.8
FLOOR_COEF = 5000.0 * 3.3228 + 10000.0          # 26614, reproduces 31.24 at E[s2]=0.001174


class FFPIPolicy:
    """ff_pi_tau driven through the torch rollout, for the controller-independence check."""

    def __init__(self, B):
        from controllers.ff_pi_tau import Controller
        self.cs = [Controller() for _ in range(B)]

    def __call__(self, ctx):
        from collections import namedtuple
        S = namedtuple('S', 'roll_lataccel v_ego a_ego')
        FP = namedtuple('FP', 'lataccel roll_lataccel v_ego a_ego')
        out = []
        fut = ctx['fut_lat'].cpu().numpy()
        for i, c in enumerate(self.cs):
            st = S(float(ctx['roll'][i]), float(ctx['v'][i]), float(ctx['a'][i]))
            fp = FP(list(fut[i]), [], [], [])
            out.append(c.update(float(ctx['target'][i]), float(ctx['cur'][i]), st, fp))
        return torch.tensor(out, dtype=torch.float32, device=DEV)


@torch.no_grad()
def sigma2(plant, segs, policy_fn, seed=0):
    """Rollout in 'sample' mode, accumulating the plant's conditional variance over the cost window."""
    torch.manual_seed(seed)
    B = len(segs)
    T = min(min(len(s['target']) for s in segs), COST_END_IDX)

    def stk(k):
        return torch.tensor(np.stack([s[k][:T] for s in segs]), dtype=torch.float32, device=DEV)

    roll, v, a, target, steer0 = stk('roll'), stk('v'), stk('a'), stk('target'), stk('steer')
    lat_hist = [target[:, i] for i in range(CONTEXT_LENGTH)]
    act_hist = [steer0[:, i] for i in range(CONTEXT_LENGTH)]
    sr = [roll[:, i] for i in range(CONTEXT_LENGTH)]
    sv = [v[:, i] for i in range(CONTEXT_LENGTH)]
    sa = [a[:, i] for i in range(CONTEXT_LENGTH)]
    cur = lat_hist[-1]
    ctrl = policy_fn(B)
    acc, n = torch.zeros(B, device=DEV), 0

    for t in range(CONTEXT_LENGTH, T):
        fp_end = min(t + FUTURE_PLAN_STEPS, T)
        ctx = dict(target=target[:, t], cur=cur, roll=roll[:, t], v=v[:, t], a=a[:, t],
                   fut_lat=target[:, t + 1:fp_end], fut_roll=roll[:, t + 1:fp_end],
                   fut_v=v[:, t + 1:fp_end], step=t)
        act = ctrl(ctx)
        if t < CONTROL_START_IDX:
            act = steer0[:, t]
        act = act.clamp(*STEER_RANGE)
        act_hist.append(act); sr.append(roll[:, t]); sv.append(v[:, t]); sa.append(a[:, t])
        st = torch.stack([torch.stack(act_hist[-CONTEXT_LENGTH:], 1),
                          torch.stack(sr[-CONTEXT_LENGTH:], 1),
                          torch.stack(sv[-CONTEXT_LENGTH:], 1),
                          torch.stack(sa[-CONTEXT_LENGTH:], 1)], dim=-1)
        past = torch.stack(lat_hist[-CONTEXT_LENGTH:], 1)
        tok = plant.tokenize(past)
        logits = plant.m(st, tok)[:, -1]
        probs = F.softmax(logits / TEMP, -1)
        mu = (probs * plant.bins).sum(-1)
        var = (probs * plant.bins ** 2).sum(-1) - mu ** 2
        if t >= CONTROL_START_IDX:
            acc = acc + var
            n += 1
        idx = torch.multinomial(probs, 1).squeeze(-1)
        pred = torch.clamp(plant.bins[idx], cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA)
        cur = pred if t >= CONTROL_START_IDX else target[:, t]
        lat_hist.append(cur)
    return (acc / max(n, 1)).cpu().numpy()


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    out = sys.argv[3] if len(sys.argv) > 3 else f'segfloor_{start}_{n}.csv'
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    files = ALL[start:start + n]
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV)
    net.load_state_dict(torch.load('cnn_v4.pt', map_location=DEV)); net.eval()

    rows = []
    CH = 40
    for i in range(0, len(files), CH):
        segs = [load_segment(f) for f in files[i:i + CH]]
        B = len(segs)
        s_cnn = sigma2(plant, segs, lambda b: AblPolicy(net, b, DEV))
        s_ffpi = sigma2(plant, segs, lambda b: FFPIPolicy(b))
        for j, f in enumerate(files[i:i + CH]):
            rows.append(dict(seg=str(f), s2_cnn=float(s_cnn[j]), s2_ffpi=float(s_ffpi[j])))
        print(f'  {i + B}/{len(files)}', flush=True)

    D = pd.DataFrame(rows)
    D['floor_noise'] = D.s2_cnn * FLOOR_COEF
    D.to_csv(out, index=False)
    r = np.corrcoef(D.s2_cnn, D.s2_ffpi)[0, 1]
    print(f'\n  per-segment sigma^2 (cnn_v4):  mean {D.s2_cnn.mean():.6f}  median {D.s2_cnn.median():.6f}  '
          f'p10 {D.s2_cnn.quantile(.10):.6f}  p90 {D.s2_cnn.quantile(.90):.6f}', flush=True)
    print(f'  population constant used before: 0.001174  -> floor 31.24', flush=True)
    print(f'  per-segment noise floor:  mean {D.floor_noise.mean():.2f}  median {D.floor_noise.median():.2f}  '
          f'p10 {D.floor_noise.quantile(.10):.2f}  p90 {D.floor_noise.quantile(.90):.2f}', flush=True)
    print(f'\n  CONTROLLER INDEPENDENCE: corr(sigma^2 under cnn_v4, under ff_pi_tau) = {r:.3f}', flush=True)
    print(f'    ratio ff_pi/cnn: median {(D.s2_ffpi / D.s2_cnn).median():.3f}', flush=True)
    print(f'  wrote {out}', flush=True)


if __name__ == '__main__':
    main()
