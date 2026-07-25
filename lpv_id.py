"""Stage A: identify an LPV-ARX model of the TinyPhysics plant by probing it directly.

Model (per v_ego bin, coefficients later smoothed in v):
  ARX(1,1):  lat[t] = a1*lat[t-1] + b1*steer[t] + c*roll[t] + d
  ARX(2,2):  lat[t] = a1*lat[t-1] + a2*lat[t-2] + b1*steer[t] + b2*steer[t-1] + c*roll[t] + d

Alignment note (verified against torch_sim.rollout): the plant's prediction for index t
consumes the state/action windows up to and including t, so lat[t] is the response to steer[t].

Data comes from the SAMPLED plant (the thing we actually control, not the Jensen-biased
expected path), from a mix of:
  - on-policy: the trained cnn controller + exploration noise (covers where MPC will operate)
  - off-policy: random-walk steering (excites the dynamics broadly, identifies the gain)
"""
import sys, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment, rollout
from nets import AblNet, AblPolicy
from tinyphysics import CONTEXT_LENGTH as CL, CONTROL_START_IDX, COST_END_IDX, STEER_RANGE

DEV = 'cuda'
ALL = sorted(Path('data/SYNTHETIC').iterdir())
V_EDGES = np.array([0, 8, 14, 18, 22, 26, 30, 34, 100], dtype=np.float32)
# identification segments are chosen in __main__ (argv is not parsed at import time, so that
# this module can be imported by other scripts)


class RandomWalk:
    """Off-policy excitation: per-segment random walk with varied step size."""
    def __init__(self, B, dev, seed=0):
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.step = (0.05 + 0.45 * torch.rand(B, generator=g)).to(dev)
        self.a = torch.zeros(B, device=dev)
        self.g = torch.Generator(device=dev).manual_seed(seed)
    def __call__(self, ctx):
        n = (torch.rand(self.a.shape, generator=self.g, device=self.a.device) * 2 - 1) * self.step
        self.a = (self.a + n).clamp(STEER_RANGE[0], STEER_RANGE[1])
        return self.a


class NoisyPolicy:
    """On-policy excitation: trained controller + exploration noise."""
    def __init__(self, net, B, dev, sigma=0.15, seed=0):
        self.inner = AblPolicy(net, B, dev); self.sigma = sigma
        self.g = torch.Generator(device=dev).manual_seed(seed)
    def __call__(self, ctx):
        a = self.inner(ctx)
        n = torch.randn(a.shape, generator=self.g, device=a.device) * self.sigma
        return (a + n).clamp(STEER_RANGE[0], STEER_RANGE[1])


def collect(plant, files, kind, net=None, bs=40, seed=0):
    """-> dict of stacked per-step arrays over the control window."""
    out = {k: [] for k in ['lat', 'lat1', 'lat2', 'st', 'st1', 'roll', 'v', 'a']}
    for i in range(0, len(files), bs):
        segs = [load_segment(f) for f in files[i:i + bs]]
        B = len(segs)
        ctrl = RandomWalk(B, DEV, seed + i) if kind == 'rw' else NoisyPolicy(net, B, DEV, seed=seed + i)
        with torch.no_grad():
            traj, _, acts = rollout(plant, segs, ctrl, mode='sample', stop=COST_END_IDX,
                                    return_actions=True)
        T = traj.shape[1]
        roll = torch.tensor(np.stack([s['roll'][:T] for s in segs]), dtype=torch.float32, device=DEV)
        v = torch.tensor(np.stack([s['v'][:T] for s in segs]), dtype=torch.float32, device=DEV)
        aeg = torch.tensor(np.stack([s['a'][:T] for s in segs]), dtype=torch.float32, device=DEV)
        lo, hi = CONTROL_START_IDX, T                       # transitions strictly inside control
        # flatten per batch: segment lengths differ, so T varies between batches
        out['lat'].append(traj[:, lo:hi].reshape(-1)); out['lat1'].append(traj[:, lo - 1:hi - 1].reshape(-1))
        out['lat2'].append(traj[:, lo - 2:hi - 2].reshape(-1))
        out['st'].append(acts[:, lo:hi].reshape(-1));  out['st1'].append(acts[:, lo - 1:hi - 1].reshape(-1))
        out['roll'].append(roll[:, lo:hi].reshape(-1)); out['v'].append(v[:, lo:hi].reshape(-1))
        out['a'].append(aeg[:, lo:hi].reshape(-1))
        print(f"  {kind}: {i + B}/{len(files)} segs (T={T})", flush=True)
    return {k: torch.cat(vs).cpu().numpy() for k, vs in out.items()}


def fit(D, order):
    """Least-squares fit per v bin. Returns coeffs [nbin, nparam], R2 [nbin], counts."""
    cols = ([D['lat1'], D['st'], D['roll'], np.ones_like(D['lat1'])] if order == 1 else
            [D['lat1'], D['lat2'], D['st'], D['st1'], D['roll'], np.ones_like(D['lat1'])])
        # ARX(1,1): a1, b1, c, d      ARX(2,2): a1, a2, b1, b2, c, d
    X = np.stack(cols, -1); y = D['lat']
    nb = len(V_EDGES) - 1
    C = np.zeros((nb, X.shape[1])); R2 = np.zeros(nb); N = np.zeros(nb, dtype=int)
    for b in range(nb):
        m = (D['v'] >= V_EDGES[b]) & (D['v'] < V_EDGES[b + 1])
        N[b] = m.sum()
        if N[b] < 500:
            C[b] = C[b - 1] if b else 0; R2[b] = np.nan; continue
        Xb, yb = X[m], y[m]
        coef, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        C[b] = coef
        R2[b] = 1 - ((yb - Xb @ coef) ** 2).sum() / ((yb - yb.mean()) ** 2).sum()
    return C, R2, N


if __name__ == '__main__':
    # identification segments: disjoint from TRAIN[2000:4000], VAL, SELECT and the clean split
    ID_SEGS = ALL[6000:6000 + int(sys.argv[1] if len(sys.argv) > 1 else 240)]
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV); net.load_state_dict(torch.load('cnn_PM.pt')); net.eval()
    half = len(ID_SEGS) // 2
    print(f"collecting identification data from {len(ID_SEGS)} segs (half on-policy, half random-walk)")
    Dp = collect(plant, ID_SEGS[:half], 'onpolicy', net=net, seed=100)
    Dr = collect(plant, ID_SEGS[half:], 'rw', seed=200)
    D = {k: np.concatenate([Dp[k], Dr[k]]) for k in Dp}
    print(f"\ndataset: {D['lat'].size} transitions | lat range [{D['lat'].min():.2f},{D['lat'].max():.2f}] "
          f"| steer range [{D['st'].min():.2f},{D['st'].max():.2f}]")

    for order in (1, 2):
        C, R2, N = fit(D, order)
        print(f"\n=== ARX({order},{order}) per v_ego bin ===")
        names = ['a1', 'b1', 'c', 'd'] if order == 1 else ['a1', 'a2', 'b1', 'b2', 'c', 'd']
        print("  v_bin      n     " + "  ".join(f"{n:>7}" for n in names) + "     R2")
        for b in range(len(V_EDGES) - 1):
            if np.isnan(R2[b]): continue
            print(f"  {V_EDGES[b]:>2.0f}-{V_EDGES[b+1]:<3.0f} {N[b]:>8} " +
                  "  ".join(f"{c:7.4f}" for c in C[b]) + f"  {R2[b]:6.4f}")
        np.savez(f'lpv_arx{order}.npz', C=C, V_EDGES=V_EDGES, R2=R2, N=N, order=order)
        print(f"  saved lpv_arx{order}.npz")
