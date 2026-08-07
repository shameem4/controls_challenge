"""Definitive checks on the two plant quantities where outside analyses disagree with ours.

Both are load-bearing: the noise magnitude anchors the causal cost floor (and hence every
"how much is recoverable" claim), and the DC gain anchors the inverse-plant feedforward that the best
classical controller is built on.

  DC GAIN.  `nurikserikbayev` reports ~2.0 from excitation-based ARX identification. Ours is
            G(v) = 0.0093v + 1.34 (about 1.5-1.7 at typical speeds), with ff_pi then applying a tuned
            gain_scale of 1.79 on top. Measured here as a true steady-state step response from
            equilibrium in `expected` mode -- the conditional mean, so no sampling noise at all.

  NOISE.    Ryan Lei's analysis uses sigma ~ 0.044. We measure E[sigma^2] = 0.001174 (sigma ~ 0.0343)
            from the plant's conditional output distribution. Three different quantities are computed
            here to find out whether the disagreement is a measurement error or a definition
            mismatch:
              (a) conditional  sqrt(E[Var]) of the output distribution, pre-clamp   <- what we use
              (b) realised     sqrt(E[(sample - conditional mean)^2]), pre-clamp
              (c) post-clamp   sqrt(E[(lataccel[t] - lataccel[t-1])^2]) innovations after MAX_ACC_DELTA
            (a) and (b) must agree by construction; (c) is a different quantity and is the most
            likely source of a larger number.

Usage: python verify_plant.py
"""
import numpy as np, torch
import torch.nn.functional as F
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy
from tinyphysics import (CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX,
                         FUTURE_PLAN_STEPS, STEER_RANGE, MAX_ACC_DELTA, ACC_G)

DEV = 'cuda'
TEMP = 0.8
GAIN_FIT = np.load(Path(__file__).resolve().parent / 'gain_fit.npy')


@torch.no_grad()
def dc_gain(plant, v, roll=0.0, a=0.0, du=0.10, warm=200, settle=150, n=8):
    """True steady-state dlataccel/dsteer at equilibrium, in expected (noise-free) mode.

    Two arms from the SAME equilibrium: hold u0, and hold u0+du. The difference of the settled
    lataccels divided by du is the DC gain. Running in `expected` mode removes sampling entirely, so
    no averaging over draws is needed.
    """
    G0 = float(np.clip(np.polyval(GAIN_FIT, v), 0.3, 4.0))
    u0 = (0.0 - roll) / G0                       # equilibrium steer for c = 0
    steps = warm + settle
    out = []
    for extra in (0.0, du):
        # explicit stepper: sysid.Rig hardcodes mode='sample', and this test needs the noise-free
        # conditional mean so a single run suffices instead of averaging over draws.
        rollv = torch.full((n, CONTEXT_LENGTH), roll, device=DEV)
        vv = torch.full((n, CONTEXT_LENGTH), v, device=DEV)
        av = torch.full((n, CONTEXT_LENGTH), a, device=DEV)
        lat = [torch.zeros(n, device=DEV) for _ in range(CONTEXT_LENGTH)]
        act = [torch.full((n,), u0, device=DEV) for _ in range(CONTEXT_LENGTH)]
        cur = lat[-1]
        for k in range(steps):
            u = torch.full((n,), u0 + (extra if k >= warm else 0.0), device=DEV)
            act.append(u)
            st = torch.stack([torch.stack(act[-CONTEXT_LENGTH:], 1), rollv, vv, av], -1)
            pred = plant.step(st, plant.tokenize(torch.stack(lat[-CONTEXT_LENGTH:], 1)), mode='expected')
            cur = torch.clamp(pred, cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA)
            lat.append(cur)
        out.append(float(cur.mean()))
    return (out[1] - out[0]) / du


@torch.no_grad()
def noise_stats(plant, files, seed=0):
    """Conditional variance, realised pre-clamp innovation, and post-clamp step, on real segments."""
    segs = [load_segment(f) for f in files]
    B = len(segs)
    T = min(min(len(s['target']) for s in segs), COST_END_IDX)
    torch.manual_seed(seed)
    net = AblNet('PM').to(DEV)
    net.load_state_dict(torch.load('cnn_v4.pt', map_location=DEV)); net.eval()

    def stk(k):
        return torch.tensor(np.stack([s[k][:T] for s in segs]), dtype=torch.float32, device=DEV)
    roll, v, a, target, steer0 = stk('roll'), stk('v'), stk('a'), stk('target'), stk('steer')
    lat = [target[:, i] for i in range(CONTEXT_LENGTH)]
    act = [steer0[:, i] for i in range(CONTEXT_LENGTH)]
    sr = [roll[:, i] for i in range(CONTEXT_LENGTH)]
    sv = [v[:, i] for i in range(CONTEXT_LENGTH)]
    sa = [a[:, i] for i in range(CONTEXT_LENGTH)]
    cur = lat[-1]
    pol = AblPolicy(net, B, DEV)
    cvar, innov, postd, nclamp, nstep = [], [], [], 0, 0

    for t in range(CONTEXT_LENGTH, T):
        fe = min(t + FUTURE_PLAN_STEPS, T)
        ctx = dict(target=target[:, t], cur=cur, roll=roll[:, t], v=v[:, t], a=a[:, t],
                   fut_lat=target[:, t + 1:fe], fut_roll=roll[:, t + 1:fe],
                   fut_v=v[:, t + 1:fe], step=t)
        u = pol(ctx)
        if t < CONTROL_START_IDX:
            u = steer0[:, t]
        u = u.clamp(*STEER_RANGE)
        act.append(u); sr.append(roll[:, t]); sv.append(v[:, t]); sa.append(a[:, t])
        st = torch.stack([torch.stack(act[-CONTEXT_LENGTH:], 1), torch.stack(sr[-CONTEXT_LENGTH:], 1),
                          torch.stack(sv[-CONTEXT_LENGTH:], 1), torch.stack(sa[-CONTEXT_LENGTH:], 1)], -1)
        logits = plant.m(st, plant.tokenize(torch.stack(lat[-CONTEXT_LENGTH:], 1)))[:, -1]
        p = F.softmax(logits / TEMP, -1)
        mu = (p * plant.bins).sum(-1)
        var = (p * plant.bins ** 2).sum(-1) - mu ** 2
        idx = torch.multinomial(p, 1).squeeze(-1)
        raw = plant.bins[idx]
        pred = torch.clamp(raw, cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA)
        if t >= CONTROL_START_IDX:
            cvar.append(var); innov.append((raw - mu) ** 2); postd.append((pred - cur) ** 2)
            nclamp += int((raw != pred).sum()); nstep += raw.numel()
        cur = pred if t >= CONTROL_START_IDX else target[:, t]
        lat.append(cur)
    f = lambda x: float(torch.cat(x).mean())
    return f(cvar), f(innov), f(postd), nclamp / max(nstep, 1)


def main():
    plant = Plant(device=DEV)
    print('=== 1. DC GAIN: steady-state step response from equilibrium, expected mode ===', flush=True)
    print(f'  {"v (m/s)":>8} {"measured":>10} {"our G(v)":>10} {"ratio":>7}', flush=True)
    for v in (8.0, 15.0, 22.0, 28.0, 34.0):
        g = dc_gain(plant, v)
        gv = float(np.clip(np.polyval(GAIN_FIT, v), 0.3, 4.0))
        print(f'  {v:8.1f} {g:10.3f} {gv:10.3f} {g / gv:7.3f}', flush=True)
    print('  (outside ARX identification reports ~2.0)', flush=True)

    print('\n=== 2. NOISE: three different sigmas on the same rollouts ===', flush=True)
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    cv, iv, pd_, cl = noise_stats(plant, ALL[5000:5200])
    print(f'  (a) conditional  E[Var]      = {cv:.6f}   sigma = {np.sqrt(cv):.4f}   <- what our floor uses',
          flush=True)
    print(f'  (b) realised     E[(x-mu)^2] = {iv:.6f}   sigma = {np.sqrt(iv):.4f}   (must match (a))',
          flush=True)
    print(f'  (c) post-clamp   E[(dc)^2]   = {pd_:.6f}   sigma = {np.sqrt(pd_):.4f}   (different quantity)',
          flush=True)
    print(f'  rate clamp binds on {100 * cl:.3f}% of steps', flush=True)
    print(f'\n  implied causal total floor = sigma^2 * 26614:', flush=True)
    for lbl, s2 in (('(a) conditional', cv), ('(c) post-clamp', pd_), ('Ryan Lei sigma=0.044', 0.044 ** 2)):
        print(f'    {lbl:24} -> {s2 * 26614:7.2f}', flush=True)
    print('  cnn_v4 scores 45.32, so any floor above that is refuted by construction.', flush=True)


if __name__ == '__main__':
    main()
