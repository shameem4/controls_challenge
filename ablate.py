"""Train an AblNet config (curriculum + K=1), save periodic checkpoints for real-sim
selection.  Usage: python ablate.py <cfg> <iters> [gumbel|expected] [warmstart.pt]
  - train_mode 'gumbel' (stochastic, default) or 'expected' (deterministic plant)
  - warmstart.pt: load weights and fine-tune (full horizon from the start)"""
import sys, os, numpy as np, torch
from torch_sim import Plant, load_segment, rollout, cost
from nets import AblNet, AblPolicy
from train import TRAIN, VAL, ALL
from tinyphysics import COST_END_IDX

DEV = 'cuda'
cfg_arg = sys.argv[1]; cfg = '' if cfg_arg == 'base' else cfg_arg
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 300
train_mode = sys.argv[3] if len(sys.argv) > 3 else 'gumbel'   # 'gumbel'/'expected' [+ '_soft' for soft-token BPTT]
init_ckpt = sys.argv[4] if len(sys.argv) > 4 else None        # warm-start weights (fine-tune)
SOFT = train_mode.endswith('_soft'); train_mode = train_mode[:-5] if SOFT else train_mode
TBPTT = int(os.environ.get('TBPTT', 30))
TRAIN_N = int(os.environ.get('TRAIN_N', 0))                   # >0 => use ALL[2000:2000+N] (bigger set, still disjoint)
if TRAIN_N: TRAIN = ALL[2000:2000 + TRAIN_N]
os.makedirs('ckpts', exist_ok=True)
plant = Plant(device=DEV)
net = AblNet(cfg).to(DEV)
SUR = None
if 'J' in cfg:
    from surrogate import Surrogate
    SUR = Surrogate().to(DEV); SUR.load_state_dict(torch.load('surrogate.pt')); SUR.eval()
    for _p in SUR.parameters(): _p.requires_grad_(False)
if init_ckpt:
    net.load_state_dict(torch.load(init_ckpt))
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
                traj, target = rollout(plant, segs, AblPolicy(net, len(segs), DEV, sur=SUR), mode='sample', stop=COST_END_IDX)
                t.append(cost(traj, target)[2])
            tots.append(torch.cat(t).mean().item())
    return float(np.mean(tots))

best = 1e9
print(f"[{cfg_arg}] params={sum(p.numel() for p in net.parameters())} iters={iters} train_mode={train_mode} soft={SOFT}", flush=True)
for it in range(iters):
    stop = COST_END_IDX if init_ckpt else min(150 + it, COST_END_IDX)   # fine-tune: full horizon from the start
    opt.zero_grad(); tl = 0.0
    for _ in range(ACC):
        idx = torch.randint(len(TRAIN), (bs,))
        segs = [load_segment(TRAIN[k]) for k in idx]
        traj, target = rollout(plant, segs, AblPolicy(net, bs, DEV, sur=SUR), mode=train_mode, tbptt=TBPTT, stop=stop, soft_tokens=SOFT)
        loss = cost(traj, target)[2].mean() / ACC
        loss.backward(); tl += loss.item()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if it % 25 == 0 or it == iters - 1:
        torch.save(net.state_dict(), f'ckpts/abl{cfg_arg}_{it:04d}.pt')
        vc = seeded_val(); tag = ''
        if vc < best: best = vc; tag = ' *'
        print(f"[{cfg_arg}] it{it:3d} stop={stop} train={tl:6.2f} torch_val={vc:6.2f}{tag}", flush=True)
print(f"[{cfg_arg}] done best_torch={best:.2f}", flush=True)
