"""Improved LPV identification: ARX(1, n_b) fitted to minimize MULTI-STEP free-run error.

Diagnosis this addresses (from lpv_floor.py): the first-order 1-step-LS model is ~2x worse than
the plant's own predictive floor at H=30, and the mismatch term dominates the noise term. Causes:
  (1) underspecification -- one steer term to approximate a transformer with a 20-step context;
  (2) 1-step-optimal fitting, which is not multi-step-optimal (ARX(2,2) even destabilised free-run).

Model, per v_ego bin:
    lat[t] = a*lat[t-1] + sum_{k=0..nb-1} b_k*steer[t-k] + sum_{k=0..nr-1} g_k*roll[t-k] + d
  - `a` is parameterised as tanh(.) so |a|<1: stable free-running by construction.
  - the steer FIR captures the input lag the first-order model was missing.
  - fitted by autograd on the H-step free-run loss, initialised from the 1-step LS solution.

Usage: python lpv_id2.py [nseg] [nb] [H]      -> writes lpv_arx_ms.npz
"""
import sys, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment, rollout
from nets import AblNet, AblPolicy
from lpv_id import RandomWalk, NoisyPolicy, V_EDGES
from tinyphysics import CONTROL_START_IDX, COST_END_IDX

DEV = 'cuda'
ALL = sorted(Path('data/SYNTHETIC').iterdir())
CACHE = Path('lpv_traj_cache.npz')


def collect_traj(plant, net, files, bs=40):
    """Collect full trajectories (not flattened) so we can fit free-run losses."""
    T_KEEP = COST_END_IDX
    out = {k: [] for k in ['lat', 'st', 'roll', 'v']}
    half = len(files) // 2
    for i in range(0, len(files), bs):
        chunk = files[i:i + bs]
        segs = [load_segment(f) for f in chunk]
        B = len(segs)
        kind = 'onpolicy' if i < half else 'rw'
        ctrl = (NoisyPolicy(net, B, DEV, seed=1000 + i) if kind == 'onpolicy'
                else RandomWalk(B, DEV, 2000 + i))
        with torch.no_grad():
            traj, _, acts = rollout(plant, segs, ctrl, mode='sample', stop=T_KEEP,
                                    return_actions=True)
        T = traj.shape[1]
        if T < T_KEEP:                        # rare short segment -> skip for uniform shapes
            print(f"  skip chunk {i} (T={T} < {T_KEEP})", flush=True); continue
        rl = np.stack([s['roll'][:T] for s in segs]); vv = np.stack([s['v'][:T] for s in segs])
        out['lat'].append(traj.cpu().numpy()); out['st'].append(acts.cpu().numpy())
        out['roll'].append(rl.astype(np.float32)); out['v'].append(vv.astype(np.float32))
        print(f"  {kind}: {i + B}/{len(files)}", flush=True)
    return {k: np.concatenate(v, 0) for k, v in out.items()}


def build_windows(D, nb, nr, H, stride=5):
    """Windows of contiguous time, binned by v at the window start.
    Returns per-bin dict of (lat0, st_win, roll_win, y) tensors."""
    lag = max(nb, nr) - 1
    t0 = CONTROL_START_IDX + lag + 1
    T = D['lat'].shape[1]
    starts = np.arange(t0, T - H, stride)
    bins = {}
    for b in range(len(V_EDGES) - 1):
        bins[b] = {'lat0': [], 'st': [], 'roll': [], 'y': []}
    for t in starts:
        vb = np.clip(np.searchsorted(V_EDGES, D['v'][:, t], side='right') - 1, 0, len(V_EDGES) - 2)
        for b in np.unique(vb):
            m = vb == b
            bins[int(b)]['lat0'].append(D['lat'][m, t - 1])
            # steer/roll windows need indices t-lag .. t+H-1
            bins[int(b)]['st'].append(D['st'][m, t - lag:t + H])
            bins[int(b)]['roll'].append(D['roll'][m, t - lag:t + H])
            bins[int(b)]['y'].append(D['lat'][m, t:t + H])
    out = {}
    for b, d in bins.items():
        if not d['lat0']:
            out[b] = None; continue
        out[b] = tuple(torch.tensor(np.concatenate(d[k], 0), dtype=torch.float32, device=DEV)
                       for k in ['lat0', 'st', 'roll', 'y'])
    return out


def free_run(a, bcoef, gcoef, d, lat0, st, roll, H, lag):
    """Differentiable H-step free-run. st/roll are [N, lag+H]; returns [N,H]."""
    nb, nr = bcoef.shape[0], gcoef.shape[0]
    preds = []
    lat = lat0
    for h in range(H):
        i = lag + h                                   # index of "current" step in the window
        u = sum(bcoef[k] * st[:, i - k] for k in range(nb))
        r = sum(gcoef[k] * roll[:, i - k] for k in range(nr))
        lat = a * lat + u + r + d
        preds.append(lat)
    return torch.stack(preds, 1)


def fit_bin(data, nb, nr, H, lag, iters=400, lr=0.02):
    lat0, st, roll, y = data
    # --- init from 1-step least squares on the same windows ---
    cols = [lat0] + [st[:, lag - k] for k in range(nb)] + [roll[:, lag - k] for k in range(nr)]
    X = torch.stack(cols + [torch.ones_like(lat0)], 1)
    sol = torch.linalg.lstsq(X, y[:, 0:1]).solution.squeeze(-1)
    a0 = float(sol[0].clamp(-0.98, 0.98))
    pa = torch.tensor(np.arctanh(a0), device=DEV, requires_grad=True)
    pb = sol[1:1 + nb].clone().detach().requires_grad_(True)
    pg = sol[1 + nb:1 + nb + nr].clone().detach().requires_grad_(True)
    pd = sol[-1].clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([pa, pb, pg, pd], lr)
    for it in range(iters):
        pred = free_run(torch.tanh(pa), pb, pg, pd, lat0, st, roll, H, lag)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        final = ((free_run(torch.tanh(pa), pb, pg, pd, lat0, st, roll, H, lag) - y) ** 2).mean().sqrt()
    return (float(torch.tanh(pa)), pb.detach().cpu().numpy(), pg.detach().cpu().numpy(),
            float(pd), float(final), lat0.shape[0])


if __name__ == '__main__':
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    H = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    nr = 3
    lag = max(nb, nr) - 1

    if CACHE.exists():
        D = {k: v for k, v in np.load(CACHE).items()}
        print(f"loaded cached trajectories: {D['lat'].shape}")
    else:
        plant = Plant(device=DEV)
        net = AblNet('PM').to(DEV); net.load_state_dict(torch.load('cnn_PM.pt')); net.eval()
        print(f"collecting trajectories from {nseg} segs")
        D = collect_traj(plant, net, ALL[6000:6000 + nseg])
        np.savez(CACHE, **D); print(f"cached -> {CACHE}  {D['lat'].shape}")

    W = build_windows(D, nb, nr, H)
    print(f"\nfitting ARX(1,nb={nb}) + roll FIR(nr={nr}), multi-step loss over H={H}")
    C = np.zeros((len(V_EDGES) - 1, 1 + nb + nr + 1)); RMS = np.full(len(V_EDGES) - 1, np.nan)
    print("  v_bin        n      a     b[0..]                                  d      freerunRMS")
    for b in range(len(V_EDGES) - 1):
        if W[b] is None or W[b][0].shape[0] < 200:
            if b: C[b] = C[b - 1]
            continue
        a, bc, gc, d, rms, n = fit_bin(W[b], nb, nr, H, lag)
        C[b] = np.concatenate([[a], bc, gc, [d]]); RMS[b] = rms
        print(f"  {V_EDGES[b]:>2.0f}-{V_EDGES[b+1]:<3.0f} {n:>8}  {a:6.4f}  " +
              " ".join(f"{x:6.3f}" for x in bc) + f"  {d:7.4f}   {rms:.4f}", flush=True)
    np.savez('lpv_arx_ms.npz', C=C, V_EDGES=V_EDGES, nb=nb, nr=nr, RMS=RMS)
    print(f"\nsaved lpv_arx_ms.npz   (steady-state gains b_sum/(1-a): " +
          ", ".join(f"{C[b,1:1+nb].sum()/(1-C[b,0]):.2f}" for b in range(len(V_EDGES)-1) if not np.isnan(RMS[b])) + ")")
