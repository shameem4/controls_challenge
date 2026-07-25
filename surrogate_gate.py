"""Gate: multi-step free-run error of the SURROGATE vs the LPV vs the plant-twin floor,
on identical branch points. This decides whether surrogate-MPC is worth building.

Reference (from lpv_floor.py, on-policy, held-out):
    H=30   LPV 0.449   plant-twin 0.220   -> LPV is 2.05x the floor
"""
import sys, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment, rollout
from nets import AblNet, AblPolicy
from surrogate import Surrogate
from lpv_validate import LPV
from lpv_floor import plant_twin_freerun
from tinyphysics import CONTEXT_LENGTH as CL, CONTROL_START_IDX, COST_END_IDX, MAX_ACC_DELTA

DEV = 'cuda'
ALL = sorted(Path('data/SYNTHETIC').iterdir())
HS = [1, 5, 10, 20, 30]


@torch.no_grad()
def surrogate_freerun(sur, traj, acts, roll, v, aeg, t, H):
    """Free-run the surrogate H steps from the true state at t. Returns lataccel at t+H-1."""
    lat_win = traj[:, t - CL:t].clone()          # lat[t-CL .. t-1]
    prev = traj[:, t - 1]
    for h in range(H):
        i = t + h
        pred = sur(acts[:, i - CL + 1:i + 1], roll[:, i - CL + 1:i + 1],
                   v[:, i - CL + 1:i + 1], aeg[:, i - CL + 1:i + 1], lat_win)
        pred = torch.clamp(pred, prev - MAX_ACC_DELTA, prev + MAX_ACC_DELTA)   # plant's slew clamp
        lat_win = torch.cat([lat_win[:, 1:], pred[:, None]], 1)
        prev = pred
    return prev


def main(nseg=20):
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV); net.load_state_dict(torch.load('cnn_PM.pt')); net.eval()
    sur = Surrogate().to(DEV); sur.load_state_dict(torch.load('surrogate.pt')); sur.eval()
    lpv = LPV('lpv_arx1.npz')

    segs = [load_segment(f) for f in ALL[7000:7000 + nseg]]
    B = len(segs)
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
    print(f"segs={B}  branch points/seg={len(branch)}   (identical points for all predictors)\n")
    print(f"{'H':>3}  {'surrogate':>10}  {'LPV':>8}  {'plant-twin':>11}   ratio(sur)  ratio(LPV)")
    for H in HS:
        e_s, e_l, e_t = [], [], []
        for t in branch:
            e_t.append(plant_twin_freerun(plant, traj, acts, roll, v, aeg, t, H).cpu().numpy()
                       - tj[:, t + H - 1])
            e_s.append(surrogate_freerun(sur, traj, acts, roll, v, aeg, t, H).cpu().numpy()
                       - tj[:, t + H - 1])
            for b in range(B):
                p = lpv.predict_h(tj[b, t - 1], tj[b, t - 2], ac[b, t - LAG:t],
                                  ac[b, t:t + H], rl[b, t - LAG:t], rl[b, t:t + H],
                                  vv[b, t:t + H], H)
                e_l.append(p[-1] - tj[b, t + H - 1])
        rs = float(np.sqrt(np.mean(np.concatenate(e_s) ** 2)))
        rl_ = float(np.sqrt(np.mean(np.array(e_l) ** 2)))
        rt = float(np.sqrt(np.mean(np.concatenate(e_t) ** 2)))
        print(f"{H:>3}  {rs:>10.4f}  {rl_:>8.4f}  {rt:>11.4f}   {rs/rt:>9.2f}  {rl_/rt:>10.2f}",
              flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
