"""Is the LPV model's multi-step error model MISMATCH, or the plant's irreducible NOISE?

Control experiment ("twin"): branch at the true state and free-run H steps with the *true plant*
(a perfect model, only fresh sampling noise). Its error is the information floor. Compare the
LPV model's error on the exact same branch points.

  LPV error >> plant-twin error  -> model mismatch, worth improving  (MPC limited by the model)
  LPV error ~= plant-twin error  -> LPV is at the noise floor, nothing to fix (MPC limited by noise)
"""
import sys, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment, rollout
from nets import AblNet, AblPolicy
from lpv_validate import LPV
from tinyphysics import CONTEXT_LENGTH as CL, CONTROL_START_IDX, MAX_ACC_DELTA, COST_END_IDX

DEV = 'cuda'
ALL = sorted(Path('data/SYNTHETIC').iterdir())
HS = [1, 5, 10, 20, 30]


@torch.no_grad()
def plant_twin_freerun(plant, traj, acts, roll, v, aeg, t, H):
    """Free-run the TRUE plant H steps from the true state at t, using known future actions.
    traj/acts/roll/v/aeg: [B,T] tensors. Returns predicted lataccel at t+H-1, shape [B]."""
    lat = [traj[:, t - CL + k] for k in range(CL)]          # lat[t-CL..t-1]
    prev = traj[:, t - 1]
    for h in range(H):
        i = t + h
        st = torch.stack([acts[:, i - CL + 1:i + 1], roll[:, i - CL + 1:i + 1],
                          v[:, i - CL + 1:i + 1], aeg[:, i - CL + 1:i + 1]], -1)
        tok = plant.tokenize(torch.stack(lat[-CL:], 1))
        pred = plant.step(st, tok, mode='sample')
        pred = torch.clamp(pred, prev - MAX_ACC_DELTA, prev + MAX_ACC_DELTA)
        lat.append(pred); prev = pred
    return prev


def main(nseg=20, models=('lpv_arx1.npz', 'lpv_arx_ms.npz')):
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV); net.load_state_dict(torch.load('cnn_PM.pt')); net.eval()
    lpvs = [(m, LPV(m)) for m in models if Path(m).exists()]
    files = ALL[7000:7000 + nseg]
    segs = [load_segment(f) for f in files]; B = len(segs)
    with torch.no_grad():
        traj, _, acts = rollout(plant, segs, AblPolicy(net, B, DEV), mode='sample',
                                stop=COST_END_IDX, return_actions=True)
    T = traj.shape[1]
    roll = torch.tensor(np.stack([s['roll'][:T] for s in segs]), dtype=torch.float32, device=DEV)
    v = torch.tensor(np.stack([s['v'][:T] for s in segs]), dtype=torch.float32, device=DEV)
    aeg = torch.tensor(np.stack([s['a'][:T] for s in segs]), dtype=torch.float32, device=DEV)
    tj, ac = traj.cpu().numpy(), acts.cpu().numpy()
    rl, vv = roll.cpu().numpy(), v.cpu().numpy()

    LAG = 8
    branch = list(range(CONTROL_START_IDX + LAG + 2, T - max(HS), 7))
    print(f"segs={B}  branch points/seg={len(branch)}  (identical points for every predictor)\n")
    hdr = "  ".join(f"{m.replace('lpv_','').replace('.npz',''):>12}" for m, _ in lpvs)
    print(f"{'H':>3}  {hdr}  {'plant-twin':>11}   ratios vs floor")
    for H in HS:
        e = {m: [] for m, _ in lpvs}; e_twin = []
        for t in branch:
            tw = plant_twin_freerun(plant, traj, acts, roll, v, aeg, t, H).cpu().numpy()
            e_twin.append(tw - tj[:, t + H - 1])
            for b in range(B):
                for m, lpv in lpvs:
                    p = lpv.predict_h(tj[b, t - 1], tj[b, t - 2], ac[b, t - LAG:t],
                                      ac[b, t:t + H], rl[b, t - LAG:t], rl[b, t:t + H],
                                      vv[b, t:t + H], H)
                    e[m].append(p[-1] - tj[b, t + H - 1])
        r_twin = float(np.sqrt(np.mean(np.concatenate(e_twin) ** 2)))
        rs = {m: float(np.sqrt(np.mean(np.array(v_) ** 2))) for m, v_ in e.items()}
        cells = "  ".join(f"{rs[m]:>12.4f}" for m, _ in lpvs)
        ratios = "  ".join(f"{m.replace('lpv_','').replace('.npz','')}={rs[m]/r_twin:.2f}" for m, _ in lpvs)
        print(f"{H:>3}  {cells}  {r_twin:>11.4f}   {ratios}", flush=True)
    print("\nratio ~1 => the linear model is as predictive as the true plant itself, i.e. the")
    print("remaining error is unpredictable drift and a better model cannot help.")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
