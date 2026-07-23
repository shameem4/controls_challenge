"""Train one ablation config of AblNet (scratch + curriculum + K=1), save periodic
checkpoints for real-sim selection.  Usage: python ablate.py <base|P|M|H|PMH> <iters>"""
import sys, os, numpy as np, torch
from torch_sim import Plant, load_segment, rollout, cost
from nets import AblNet, AblPolicy
from train import TRAIN, VAL
from tinyphysics import COST_END_IDX

DEV = 'cuda'
cfg_arg = sys.argv[1]; cfg = '' if cfg_arg == 'base' else cfg_arg
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 300
os.makedirs('ckpts', exist_ok=True)
plant = Plant(device=DEV)
net = AblNet(cfg).to(DEV)
opt = torch.optim.Adam(net.parameters(), 2e-4)
bs, ACC = 8, 4

def seeded_val(seeds=(0, 1)):
    tots = []
    for s in seeds:
        torch.manual_seed(s)
        with torch.no_grad():
            t = []
            for i in range(0, 60, bs):
                segs = [load_segment(f) for f in VAL[:60][i:i + bs]]
                traj, target = rollout(plant, segs, AblPolicy(net, len(segs), DEV), mode='sample', stop=COST_END_IDX)
                t.append(cost(traj, target)[2])
            tots.append(torch.cat(t).mean().item())
    return float(np.mean(tots))

best = 1e9
print(f"[{cfg_arg}] params={sum(p.numel() for p in net.parameters())} iters={iters}", flush=True)
for it in range(iters):
    stop = min(150 + it, COST_END_IDX)
    opt.zero_grad(); tl = 0.0
    for _ in range(ACC):
        idx = torch.randint(len(TRAIN), (bs,))
        segs = [load_segment(TRAIN[k]) for k in idx]
        traj, target = rollout(plant, segs, AblPolicy(net, bs, DEV), mode='gumbel', tbptt=30, stop=stop)
        loss = cost(traj, target)[2].mean() / ACC
        loss.backward(); tl += loss.item()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if it % 25 == 0 or it == iters - 1:
        torch.save(net.state_dict(), f'ckpts/abl{cfg_arg}_{it:04d}.pt')
        vc = seeded_val(); tag = ''
        if vc < best: best = vc; tag = ' *'
        print(f"[{cfg_arg}] it{it:3d} stop={stop} train={tl:6.2f} torch_val={vc:6.2f}{tag}", flush=True)
print(f"[{cfg_arg}] done best_torch={best:.2f}", flush=True)
