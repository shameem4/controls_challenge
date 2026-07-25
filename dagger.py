"""DAgger iteration: retrain the surrogate on the MPC's OWN state distribution.

Why: with the planning clamp fixed, the MPC is near-optimal against its model (6.17 on the
surrogate, ~ the analytic optimum) but only ~50 on the real plant, and it still needs heavy move
suppression (rdu=1e4) to behave. That gap is model exploitation -- the surrogate was fitted on
cnn+noise and random-walk data, so the aggressive states the MPC actually visits are off
distribution, and the optimiser finds plans the surrogate likes but reality punishes.

Fix: run the MPC on the real plant, query the plant's exact conditional mean at the states it
visits, add those to the training set, refit. Then the MPC can be de-tuned less and should
approach its in-model performance.

Usage: python dagger.py collect [nseg]    -> surrogate_data_mpc.npz
       python dagger.py train  [epochs]   -> refits surrogate.pt on ALL surrogate_data*.npz
"""
import sys, glob, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment, rollout
from surrogate import Surrogate, DATA, CKPT
from mpc_gn import GaussNewtonMPC
from tinyphysics import CONTEXT_LENGTH as CL, COST_END_IDX

DEV = 'cuda'
ALL = sorted(Path('data/SYNTHETIC').iterdir())
MPC_DATA = Path('surrogate_data_mpc.npz')


def collect(nseg=60, H=30, iters=3, r_du=10000.0, bs=20, off=6500):
    plant = Plant(device=DEV)
    sur = Surrogate().to(DEV); sur.load_state_dict(torch.load(CKPT)); sur.eval()
    for p in sur.parameters():
        p.requires_grad_(False)
    S, R, V, A, L, Y = [], [], [], [], [], []
    files = ALL[off:off + nseg]
    for i in range(0, len(files), bs):
        segs = [load_segment(f) for f in files[i:i + bs]]
        B = len(segs)
        ctrl = GaussNewtonMPC(sur, B, DEV, H=H, iters=iters, r_du=r_du, clamp='none')
        torch.manual_seed(1234 + i)
        traj, _, acts = rollout(plant, segs, ctrl, mode='sample', stop=COST_END_IDX,
                                return_actions=True)
        T = traj.shape[1]
        if T < COST_END_IDX:
            continue
        roll = torch.tensor(np.stack([s['roll'][:T] for s in segs]), dtype=torch.float32, device=DEV)
        v = torch.tensor(np.stack([s['v'][:T] for s in segs]), dtype=torch.float32, device=DEV)
        aeg = torch.tensor(np.stack([s['a'][:T] for s in segs]), dtype=torch.float32, device=DEV)
        for t in range(CL, T):
            st_w = acts[:, t - CL + 1:t + 1]; rl_w = roll[:, t - CL + 1:t + 1]
            v_w = v[:, t - CL + 1:t + 1]; a_w = aeg[:, t - CL + 1:t + 1]
            lat_w = traj[:, t - CL:t]
            with torch.no_grad():
                y = plant.step(torch.stack([st_w, rl_w, v_w, a_w], -1),
                               plant.tokenize(lat_w), mode='expected')
            S.append(st_w.cpu()); R.append(rl_w.cpu()); V.append(v_w.cpu())
            A.append(a_w.cpu()); L.append(lat_w.cpu()); Y.append(y.cpu())
        print(f"  MPC-data: {i + B}/{len(files)} segs", flush=True)
    d = {k: torch.cat(x).numpy().astype(np.float32) for k, x in
         zip(['steer', 'roll', 'v', 'a', 'lat', 'y'], [S, R, V, A, L, Y])}
    np.savez(MPC_DATA, **d)
    print(f"saved {MPC_DATA}  n={d['y'].size}")


def train(epochs=250):
    files = sorted(glob.glob('surrogate_data*.npz'))
    parts = [np.load(f) for f in files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in ['steer', 'roll', 'v', 'a', 'lat', 'y']}
    for f, p in zip(files, parts):
        print(f"  {f}: n={p['y'].size}")
    X = {k: torch.tensor(d[k], device=DEV) for k in ['steer', 'roll', 'v', 'a', 'lat']}
    y = torch.tensor(d['y'], device=DEV)
    n = y.shape[0]
    idx = torch.randperm(n, device=DEV); ntr = int(n * 0.95)
    tr, va = idx[:ntr], idx[ntr:]
    model = Surrogate().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 1024
    base = (y[va] - X['lat'][va][:, -1]).pow(2).mean().sqrt().item()
    print(f"combined n={n}  persistence baseline={base:.5f}")
    for ep in range(epochs):
        perm = tr[torch.randperm(tr.numel(), device=DEV)]
        model.train()
        for j in range(0, perm.numel(), bs):
            b = perm[j:j + bs]
            loss = (model(X['steer'][b], X['roll'][b], X['v'][b], X['a'][b], X['lat'][b]) - y[b]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                vr = (model(X['steer'][va], X['roll'][va], X['v'][va], X['a'][va],
                            X['lat'][va]) - y[va]).pow(2).mean().sqrt().item()
            print(f"  ep {ep:3d}  val_rmse={vr:.5f}", flush=True)
    torch.save(model.state_dict(), CKPT)
    print(f"saved {CKPT}")


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'collect':
        collect(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    elif cmd == 'train':
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 250)
