"""Stage B: GO/NO-GO gate — how well does the LPV-ARX model predict the REAL plant
multiple steps ahead? MPC quality is bounded by this, so measure it before building the QP.

For each held-out segment we replay a real controller's actions through the true plant, then
ask the linear model to predict the lataccel trajectory H steps ahead from the true state,
using only the (known) future actions/roll/v. Reported as RMS error in lataccel units.

Reference scales: the plant's own stochastic spread is ~0.19/step (measured), targets span
roughly [-4,4], and the cost weights (actual-target)^2 * 50, so a 30-step error of ~0.1 is
excellent, ~0.3 is marginal, >0.5 means MPC cannot plan usefully.
"""
import sys, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment, rollout
from nets import AblNet, AblPolicy
from tinyphysics import CONTROL_START_IDX, COST_END_IDX

DEV = 'cuda'
ALL = sorted(Path('data/SYNTHETIC').iterdir())
HORIZONS = [1, 5, 10, 20, 30]


class LPV:
    """LPV predictor. Handles both the 1-step-LS ARX(n,n) models (lpv_arx{1,2}.npz) and the
    multi-step-fitted ARX(1,nb)+roll-FIR model (lpv_arx_ms.npz)."""
    def __init__(self, path):
        z = np.load(path)
        self.C, self.V = z['C'], z['V_EDGES']
        self.ms = 'nb' in z
        if self.ms:
            self.nb, self.nr = int(z['nb']), int(z['nr'])
            self.lag = max(self.nb, self.nr) - 1
        else:
            self.order = int(z['order'])
            self.lag = 1

    def bin(self, v):
        return np.clip(np.searchsorted(self.V, v, side='right') - 1, 0, len(self.V) - 2)

    def predict_h(self, lat1, lat2, st_hist, acts, roll_hist, roll, v, H):
        """Free-run H steps.
        acts/roll/v cover t..t+H-1; st_hist/roll_hist cover t-lag..t-1 (oldest first)."""
        preds = np.empty(H)
        if self.ms:
            st = np.concatenate([st_hist[-self.lag:], acts]) if self.lag else acts
            rl = np.concatenate([roll_hist[-self.lag:], roll]) if self.lag else roll
            lat = lat1
            for h in range(H):
                c = self.C[self.bin(v[h])]
                a, b, g, d = c[0], c[1:1 + self.nb], c[1 + self.nb:1 + self.nb + self.nr], c[-1]
                i = self.lag + h
                lat = a * lat + sum(b[k] * st[i - k] for k in range(self.nb)) \
                              + sum(g[k] * rl[i - k] for k in range(self.nr)) + d
                preds[h] = lat
            return preds
        st_prev = st_hist[-1]
        for h in range(H):
            c = self.C[self.bin(v[h])]
            s, s1 = acts[h], (st_prev if h == 0 else acts[h - 1])
            p = (c[0] * lat1 + c[1] * s + c[2] * roll[h] + c[3] if self.order == 1
                 else c[0] * lat1 + c[1] * lat2 + c[2] * s + c[3] * s1 + c[4] * roll[h] + c[5])
            preds[h] = p
            lat2, lat1 = lat1, p
        return preds


def evaluate(model_path, files, plant, net):
    lpv = LPV(model_path)
    err = {h: [] for h in HORIZONS}
    for i in range(0, len(files), 40):
        segs = [load_segment(f) for f in files[i:i + 40]]
        B = len(segs)
        with torch.no_grad():
            traj, _, acts = rollout(plant, segs, AblPolicy(net, B, DEV), mode='sample',
                                    stop=COST_END_IDX, return_actions=True)
        traj = traj.cpu().numpy(); acts = acts.cpu().numpy()
        T = traj.shape[1]
        for b, s in enumerate(segs):
            roll, v = s['roll'][:T], s['v'][:T]
            for t in range(CONTROL_START_IDX + 2, T - max(HORIZONS)):
                for H in HORIZONS:
                    p = lpv.predict_h(traj[b, t - 1], traj[b, t - 2], acts[b, t - 1],
                                      acts[b, t:t + H], roll[t:t + H], v[t:t + H], H)
                    err[H].append(p[-1] - traj[b, t + H - 1])
    return {h: float(np.sqrt(np.mean(np.array(e) ** 2))) for h, e in err.items()}


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    files = ALL[7000:7000 + n]          # held out from identification (6000-6300) and everything else
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV); net.load_state_dict(torch.load('cnn_PM.pt')); net.eval()
    print("multi-step prediction RMS error (lataccel units), on-policy actions, held-out segs")
    print(f"{'model':<12} " + "  ".join(f"H={h:<5}" for h in HORIZONS))
    for mp in [p for p in ['lpv_arx1.npz', 'lpv_arx2.npz'] if Path(p).exists()]:
        e = evaluate(mp, files, plant, net)
        print(f"{mp:<12} " + "  ".join(f"{e[h]:<7.4f}" for h in HORIZONS), flush=True)
    print("\nreference: plant's own per-step stochastic spread ~0.19; targets span ~[-4,4]")
