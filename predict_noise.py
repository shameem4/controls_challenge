"""Can the plant's realised noise be PREDICTED from causal information?

This is the load-bearing question behind every seed-aware result in the project. The exploit's whole
advantage is knowing the realised innovation

    eps_t = c_t - mu_t          c_t sampled, mu_t the conditional mean

`FINDINGS_SEED_ORACLE.md` showed the seed-aware optimum reaches a tracking cost of 17.75, BELOW the
causal lataccel floor of 19.50 -- it is pre-compensating noise it can see. If any part of eps were
predictable from causal observables, a real controller could pre-compensate too, and the payoff would
be large: the causal floor is `sigma^2 * 26614`, so predicting a fraction rho of the innovation
variance moves the floor to `(1 - rho) * sigma^2 * 26614`. Even rho = 0.1 is ~3 points.

Two routes must be kept apart:

  PHYSICS   predicting eps from past innovations, state, action, conditional variance. This is
            legitimate control and is what this script measures.
  RNG       inferring the draw stream. `tinyphysics.py:116` seeds from md5(path) % 10^4, so only
            10,000 streams exist and enough observations identify which. That is segment
            fingerprinting -- a lookup table, not a controller -- and is already documented as the
            mechanism behind the sub-30 entries.

Predictors, increasingly strong, all evaluated on held-out steps:
  * autocorrelation of eps at lags 1..10
  * ridge regression on lagged eps + state features
  * a small MLP on the same features

Usage: python predict_noise.py [nseg]
"""
import sys, numpy as np, torch
import torch.nn.functional as F
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy
from tinyphysics import (CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX, FUTURE_PLAN_STEPS,
                         STEER_RANGE, MAX_ACC_DELTA)

DEV = 'cuda'
TEMP = 0.8
NLAG = 10


@torch.no_grad()
def collect(plant, segs, net, seed=0):
    """Roll out and record, per step: the innovation and the causal features available before it."""
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
    eps_hist = [torch.zeros(B, device=DEV) for _ in range(NLAG)]
    X, Y = [], []
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
        # skew of the conditional distribution: if the sampler had any bias, it would show here
        skew = (p * (plant.bins - mu.unsqueeze(-1)) ** 3).sum(-1) / (var ** 1.5 + 1e-9)
        idx = torch.multinomial(p, 1).squeeze(-1)
        raw = plant.bins[idx]
        eps = raw - mu                                   # the innovation, pre-clamp
        if t >= CONTROL_START_IDX + NLAG:
            feats = torch.stack(eps_hist[-NLAG:] + [cur, target[:, t], v[:, t], roll[:, t], a[:, t],
                                                    u, mu, var, skew], -1)
            X.append(feats.cpu()); Y.append(eps.cpu())
        eps_hist.append(eps)
        pred = torch.clamp(raw, cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA)
        cur = pred if t >= CONTROL_START_IDX else target[:, t]
        lat.append(cur)
    return torch.cat(X).numpy(), torch.cat(Y).numpy()


def r2(y, yh):
    return 1.0 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV)
    net.load_state_dict(torch.load('cnn_v4.pt', map_location=DEV)); net.eval()
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    segs = [load_segment(f) for f in ALL[5000:5000 + nseg]]
    X, Y = collect(plant, segs, net)
    n = len(Y); ntr = int(n * 0.7)
    print(f'  {n} samples, {X.shape[1]} features; sigma(eps) = {Y.std():.5f}', flush=True)

    print('\n=== 1. autocorrelation of the innovation ===', flush=True)
    for lag in (1, 2, 3, 5, 10):
        c = np.corrcoef(Y[:-lag], Y[lag:])[0, 1]
        print(f'    lag {lag:2}: {c:+.4f}', flush=True)

    print('\n=== 2. ridge regression on lagged eps + state ===', flush=True)
    Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    mu_, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = (Xtr - mu_) / sd_; Bx = (Xte - mu_) / sd_
    for lam in (1.0, 10.0, 100.0):
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ytr - Ytr.mean()))
        print(f'    lambda {lam:6.1f}: train R2 {r2(Ytr, A @ w + Ytr.mean()):+.5f}   '
              f'held-out R2 {r2(Yte, Bx @ w + Ytr.mean()):+.5f}', flush=True)

    print('\n=== 3. small MLP on the same features ===', flush=True)
    dev = DEV
    xt = torch.tensor(A, dtype=torch.float32, device=dev)
    yt = torch.tensor(Ytr - Ytr.mean(), dtype=torch.float32, device=dev)
    xv = torch.tensor(Bx, dtype=torch.float32, device=dev)
    mlp = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(mlp.parameters(), 1e-3)
    for it in range(1500):
        opt.zero_grad()
        i = torch.randint(0, len(xt), (4096,), device=dev)
        loss = F.mse_loss(mlp(xt[i]).squeeze(-1), yt[i])
        loss.backward(); opt.step()
    with torch.no_grad():
        pr = mlp(xv).squeeze(-1).cpu().numpy() + Ytr.mean()
        prt = mlp(xt).squeeze(-1).cpu().numpy() + Ytr.mean()
    print(f'    train R2 {r2(Ytr, prt):+.5f}   held-out R2 {r2(Yte, pr):+.5f}', flush=True)

    best = max(0.0, r2(Yte, pr))
    print(f'\n  best held-out R2 = {best:.5f}', flush=True)
    print(f'  a predictor with R2 = rho moves the causal floor from sigma^2*26614 to (1-rho) of it:',
          flush=True)
    s2 = float((Y ** 2).mean())
    print(f'    sigma^2 = {s2:.6f} -> floor {s2 * 26614:.2f};  at this R2 -> {s2 * 26614 * (1 - best):.2f}'
          f'  (saving {s2 * 26614 * best:.2f})', flush=True)


if __name__ == '__main__':
    main()
