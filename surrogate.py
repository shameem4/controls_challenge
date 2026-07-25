"""Distill a smooth, deterministic surrogate of TinyPhysics' MEAN dynamics.

Why this can work where the linear model could not: the plant's conditional mean is an exact,
deterministic, smooth function of its 20-step context, and we can *query it directly*
(`mode='expected'` returns sum_i p_i * bin_i). So training targets are NOISE-FREE -- this is pure
function approximation, not learning through stochasticity.

And why it can work where MPC on the raw plant could not: the raw plant is chaotic to plan against
because of the stochastic token sampling and the discrete embedding bottleneck. A smooth MLP
regressor has neither, so gradients through it are usable for planning.

The surrogate predicts the DELTA (lat_next - lat_now), which is small and better conditioned.

Usage:  python surrogate.py collect [nseg]     -> surrogate_data.npz
        python surrogate.py train [epochs]     -> surrogate.pt
"""
import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch_sim import Plant, load_segment, rollout
from nets import AblNet, AblPolicy
from lpv_id import RandomWalk, NoisyPolicy
from tinyphysics import CONTEXT_LENGTH as CL, CONTROL_START_IDX, COST_END_IDX

DEV = 'cuda'
ALL = sorted(Path('data/SYNTHETIC').iterdir())
DATA = Path('surrogate_data.npz')
CKPT = Path('surrogate.pt')
V_SC, A_SC = 30.0, 4.0          # input normalisation


class Surrogate(nn.Module):
    """Context (20 steps of steer/roll/v/a + 20 past lataccels) -> mean delta-lataccel."""
    def __init__(self, hidden=384, ctx=CL):
        super().__init__()
        self.ctx = ctx
        self.net = nn.Sequential(
            nn.Linear(ctx * 5 + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def features(self, steer, roll, v, a, lat):
        """All args [B, CL] -> [B, CL*5+1]. The lataccel history is centred on the latest value
        (so the net sees the *shape* of recent history) but the absolute level is passed
        explicitly as well -- the plant's dynamics depend on it (e.g. tyre saturation), so
        centring alone would discard necessary information."""
        lat_now = lat[:, -1:]
        return torch.cat([steer, roll, v / V_SC, a / A_SC, lat - lat_now, lat_now], -1)

    def forward(self, steer, roll, v, a, lat):
        """-> predicted next lataccel (absolute)."""
        return lat[:, -1] + self.net(self.features(steer, roll, v, a, lat)).squeeze(-1)


# ----------------------------------------------------------------------------- collection
def collect(nseg):
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV); net.load_state_dict(torch.load('cnn_PM.pt')); net.eval()
    files = ALL[6000:6000 + nseg]
    half = len(files) // 2
    S, R, V, A, L, Y = [], [], [], [], [], []
    bs = 40
    for i in range(0, len(files), bs):
        segs = [load_segment(f) for f in files[i:i + bs]]
        B = len(segs)
        ctrl = (NoisyPolicy(net, B, DEV, sigma=0.2, seed=7000 + i) if i < half
                else RandomWalk(B, DEV, 8000 + i))
        with torch.no_grad():
            traj, _, acts = rollout(plant, segs, ctrl, mode='sample', stop=COST_END_IDX,
                                    return_actions=True)
        T = traj.shape[1]
        if T < COST_END_IDX:
            continue
        roll = torch.tensor(np.stack([s['roll'][:T] for s in segs]), dtype=torch.float32, device=DEV)
        v = torch.tensor(np.stack([s['v'][:T] for s in segs]), dtype=torch.float32, device=DEV)
        aeg = torch.tensor(np.stack([s['a'][:T] for s in segs]), dtype=torch.float32, device=DEV)
        # for every step t, the plant's input window and its EXACT conditional mean output
        for t in range(CL, T):
            st_w = acts[:, t - CL + 1:t + 1]
            rl_w = roll[:, t - CL + 1:t + 1]
            v_w = v[:, t - CL + 1:t + 1]
            a_w = aeg[:, t - CL + 1:t + 1]
            lat_w = traj[:, t - CL:t]
            with torch.no_grad():
                states = torch.stack([st_w, rl_w, v_w, a_w], -1)
                tok = plant.tokenize(lat_w)
                y = plant.step(states, tok, mode='expected')       # noise-free target
            S.append(st_w.cpu()); R.append(rl_w.cpu()); V.append(v_w.cpu())
            A.append(a_w.cpu()); L.append(lat_w.cpu()); Y.append(y.cpu())
        print(f"  collected {i + B}/{len(files)} segs", flush=True)
    d = {k: torch.cat(x).numpy().astype(np.float32) for k, x in
         zip(['steer', 'roll', 'v', 'a', 'lat', 'y'], [S, R, V, A, L, Y])}
    np.savez(DATA, **d)
    print(f"saved {DATA}  n={d['y'].size}")


# ----------------------------------------------------------------------------- training
def train(epochs):
    d = np.load(DATA)
    X = {k: torch.tensor(d[k], device=DEV) for k in ['steer', 'roll', 'v', 'a', 'lat']}
    y = torch.tensor(d['y'], device=DEV)
    n = y.shape[0]
    idx = torch.randperm(n, device=DEV)
    ntr = int(n * 0.95)
    tr, va = idx[:ntr], idx[ntr:]
    model = Surrogate().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 1024      # small batch: the earlier bs=8192 gave only ~16 steps/epoch and badly undertrained
    # baseline: predicting "no change" (delta=0)
    base = (y[va] - X['lat'][va][:, -1]).pow(2).mean().sqrt().item()
    print(f"n={n}  train={ntr}  val={n-ntr}   persistence baseline RMSE={base:.5f}")
    for ep in range(epochs):
        perm = tr[torch.randperm(tr.numel(), device=DEV)]
        tot = 0.0
        model.train()
        for j in range(0, perm.numel(), bs):
            b = perm[j:j + bs]
            pred = model(X['steer'][b], X['roll'][b], X['v'][b], X['a'][b], X['lat'][b])
            loss = (pred - y[b]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * b.numel()
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = model(X['steer'][va], X['roll'][va], X['v'][va], X['a'][va], X['lat'][va])
                vr = (pv - y[va]).pow(2).mean().sqrt().item()
            print(f"  ep {ep:3d}  train_rmse={np.sqrt(tot/perm.numel()):.5f}  val_rmse={vr:.5f}"
                  f"  ({base/vr:.0f}x better than persistence)", flush=True)
    torch.save(model.state_dict(), CKPT)
    print(f"saved {CKPT}")


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'collect':
        collect(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    elif cmd == 'train':
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
