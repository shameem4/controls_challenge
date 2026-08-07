"""Is the plant's output VARIANCE controllable, or only its mean?

Every floor in this project rests on treating sigma^2 as an exogenous constant: the causal cost floor
is `sigma^2 * 26614` (31.24 at the population E[sigma^2] = 0.001174), and `segfloor.py` makes it
per-segment. That assumes the controller can move the conditional MEAN but not the conditional SPREAD.

The plant is an autoregressive transformer conditioned on 20 steps of (action, roll, v, a) and 20
quantised lataccel tokens. The action is not a scalar knob that shifts an output -- it is an entry in
the context that conditions the whole next distribution. So the assumption is worth checking directly:

  (a) at a FIXED context, sweep u and record the conditional mean and variance of the output
      distribution. If Var moves with u, there is a lever nobody here has used.
  (b) does the SHAPE of recent action history matter? Compare a smooth action history against a
      jittered one with the same mean. If jitter raises Var, smoothness is not just a jerk-cost
      virtue -- it buys a quieter plant.

Both are measured on the exact conditional distribution (sum p_i b_i^2 - (sum p_i b_i)^2), so there is
no sampling noise in the estimate at all.

Usage: python probe_variance.py [nseg]
"""
import sys, numpy as np, torch
import torch.nn.functional as F
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy
from tinyphysics import (CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX, STEER_RANGE)
from oracle_build import Stepper

DEV = 'cuda'
TEMP = 0.8


def dist_stats(plant, st, tok):
    logits = plant.m(st, tok)[:, -1]
    p = F.softmax(logits / TEMP, -1)
    mu = (p * plant.bins).sum(-1)
    var = (p * plant.bins ** 2).sum(-1) - mu ** 2
    return mu, var


@torch.no_grad()
def collect(plant, segs, net, nsamp=40, seed=0):
    """Snapshot real contexts, then probe them with modified action histories."""
    B, T = len(segs), COST_END_IDX
    torch.manual_seed(seed)
    S = Stepper(plant, segs, T)
    pol = AblPolicy(net, B, DEV)
    rng = np.random.default_rng(0)
    times = set(rng.choice(np.arange(CONTROL_START_IDX + 25, T - 5), nsamp, replace=False).tolist())

    US = np.linspace(-0.6, 0.6, 13)        # offset applied to the CURRENT action only
    JIT = [0.0, 0.02, 0.05, 0.10, 0.20]    # zero-mean jitter applied to the PAST 19 actions
    sweep_mu = np.zeros((len(US),)); sweep_var = np.zeros((len(US),))
    jit_var = np.zeros((len(JIT),)); jit_mu = np.zeros((len(JIT),))
    n = 0

    while S.t < T:
        t = S.t
        u = pol(S.ctx())
        if t in times:
            # rebuild the exact model input for this context
            acts = list(S.act[-CONTEXT_LENGTH:])
            st_roll = torch.stack([S.roll[:, i] for i in range(t - CONTEXT_LENGTH + 1, t + 1)], 1)
            st_v = torch.stack([S.v[:, i] for i in range(t - CONTEXT_LENGTH + 1, t + 1)], 1)
            st_a = torch.stack([S.a[:, i] for i in range(t - CONTEXT_LENGTH + 1, t + 1)], 1)
            past = torch.stack(S.lat[-CONTEXT_LENGTH:], 1)
            tok = plant.tokenize(past)
            base_acts = torch.stack(acts[-CONTEXT_LENGTH:], 1)     # [B, 20]

            # (a) sweep the CURRENT action
            for k, du in enumerate(US):
                aa = base_acts.clone()
                aa[:, -1] = (aa[:, -1] + float(du)).clamp(*STEER_RANGE)
                st = torch.stack([aa, st_roll, st_v, st_a], -1)
                mu, var = dist_stats(plant, st, tok)
                sweep_mu[k] += float(mu.mean()); sweep_var[k] += float(var.mean())

            # (b) jitter the PAST actions, zero-mean, current action untouched
            g = torch.Generator(device=DEV); g.manual_seed(t)
            for k, amp in enumerate(JIT):
                aa = base_acts.clone()
                if amp > 0:
                    j = (torch.rand(B, CONTEXT_LENGTH - 1, device=DEV, generator=g) * 2 - 1) * amp
                    j = j - j.mean(1, keepdim=True)             # zero-mean: same average steer
                    aa[:, :-1] = (aa[:, :-1] + j).clamp(*STEER_RANGE)
                st = torch.stack([aa, st_roll, st_v, st_a], -1)
                mu, var = dist_stats(plant, st, tok)
                jit_var[k] += float(var.mean()); jit_mu[k] += float(mu.mean())
            n += 1
        S.step(u)
    return US, sweep_mu / n, sweep_var / n, JIT, jit_mu / n, jit_var / n, n


def main():
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV)
    net.load_state_dict(torch.load('cnn_v4.pt', map_location=DEV)); net.eval()
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    segs = [load_segment(f) for f in ALL[5000:5000 + nseg]]
    US, smu, svar, JIT, jmu, jvar, n = collect(plant, segs, net)

    print(f'=== (a) sweep the CURRENT action at fixed context ({n} contexts x {nseg} segs) ===')
    print(f'  {"du":>7} {"E[mean]":>10} {"E[Var]":>11} {"sigma":>8} {"floor=Var*26614":>16}')
    for du, m, v in zip(US, smu, svar):
        print(f'  {du:+7.2f} {m:10.4f} {v:11.6f} {np.sqrt(v):8.4f} {v * 26614:16.2f}')
    i0 = int(np.argmin(np.abs(US)))
    print(f'\n  Var at du=0: {svar[i0]:.6f};  min over sweep: {svar.min():.6f} at du={US[int(np.argmin(svar))]:+.2f}')
    print(f'  relative spread of Var across the sweep: {100*(svar.max()-svar.min())/svar[i0]:.1f}%')
    dmu = np.gradient(smu, US)
    print(f'  d(mean)/du at du=0: {dmu[i0]:.4f}   (this is the local plant gain)')

    print(f'\n=== (b) zero-mean JITTER on the past 19 actions (same average steer) ===')
    print(f'  {"amp":>7} {"E[mean]":>10} {"E[Var]":>11} {"vs amp=0":>10}')
    for a, m, v in zip(JIT, jmu, jvar):
        print(f'  {a:7.2f} {m:10.4f} {v:11.6f} {100*(v/jvar[0]-1):+9.1f}%')
    print(f'\n  If jitter raises Var, a smooth action history buys a quieter plant --')
    print(f'  a mechanism no controller here has used, since we only ever shaped the mean.')


if __name__ == '__main__':
    main()
