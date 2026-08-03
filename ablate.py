"""Train an AblNet config (curriculum + K=1), save periodic checkpoints for real-sim
selection.  Usage: python ablate.py <cfg> <iters> [gumbel|expected] [warmstart.pt]
  - train_mode 'gumbel' (stochastic, default) or 'expected' (deterministic plant)
  - warmstart.pt: load weights and fine-tune (full horizon from the start)"""
import sys, os, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment, rollout, cost
from nets import AblNet, AblPolicy
from train import TRAIN, VAL, ALL
from tinyphysics import COST_END_IDX

DEV = 'cuda'
# SEED fixes both the weight init and the batch order, so an A/B between two cfgs is PAIRED --
# the cfg flag becomes the only difference instead of one sample from a noisy training process.
SEED = os.environ.get('SEED')
if SEED is not None:
    torch.manual_seed(int(SEED)); np.random.seed(int(SEED))
cfg_arg = sys.argv[1]; cfg = '' if cfg_arg == 'base' else cfg_arg
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 300
train_mode = sys.argv[3] if len(sys.argv) > 3 else 'gumbel'   # 'gumbel'/'expected' [+ '_soft' for soft-token BPTT]
init_ckpt = sys.argv[4] if len(sys.argv) > 4 else None        # warm-start weights (fine-tune)
SOFT = train_mode.endswith('_soft'); train_mode = train_mode[:-5] if SOFT else train_mode
TBPTT = int(os.environ.get('TBPTT', 30))
TRAIN_N = int(os.environ.get('TRAIN_N', 0))                   # >0 => use ALL[2000:2000+N] (bigger set, still disjoint)
if TRAIN_N: TRAIN = ALL[2000:2000 + TRAIN_N]
# TRAIN_LIST: path to a newline-delimited file of segment paths, for training on an arbitrary
# hand-picked subset. Whatever it points at is the ENTIRE training set, so anything in it is
# burned for evaluation -- pick the holdout accordingly.
TRAIN_LIST = os.environ.get('TRAIN_LIST')
if TRAIN_LIST: TRAIN = [Path(l) for l in open(TRAIN_LIST).read().split() if l]
# TAG namespaces the checkpoints. Without it two runs of the same cfg overwrite each other's
# checkpoints, which has already destroyed one control arm in this project.
TAG = os.environ.get('TAG', cfg_arg)
os.makedirs('ckpts', exist_ok=True)
plant = Plant(device=DEV)
net = AblNet(cfg).to(DEV)
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
                traj, target = rollout(plant, segs, AblPolicy(net, len(segs), DEV), mode='sample', stop=COST_END_IDX)
                t.append(cost(traj, target)[2])
            tots.append(torch.cat(t).mean().item())
    return float(np.mean(tots))

best = 1e9
print(f"[{TAG}] n_train={len(TRAIN)} params={sum(p.numel() for p in net.parameters())} iters={iters} train_mode={train_mode} soft={SOFT}", flush=True)
for it in range(iters):
    stop = COST_END_IDX if init_ckpt else min(150 + it, COST_END_IDX)   # fine-tune: full horizon from the start
    opt.zero_grad(); tl = 0.0
    for _ in range(ACC):
        idx = torch.randint(len(TRAIN), (bs,))
        segs = [load_segment(TRAIN[k]) for k in idx]
        traj, target = rollout(plant, segs, AblPolicy(net, bs, DEV), mode=train_mode, tbptt=TBPTT, stop=stop, soft_tokens=SOFT)
        loss = cost(traj, target)[2].mean() / ACC
        loss.backward(); tl += loss.item()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if it % 25 == 0 or it == iters - 1:
        torch.save(net.state_dict(), f'ckpts/abl{TAG}_{it:04d}.pt')
        vc = seeded_val(); tag = ''
        if vc < best: best = vc; tag = ' *'
        print(f"[{TAG}] it{it:3d} stop={stop} train={tl:6.2f} torch_val={vc:6.2f}{tag}", flush=True)
print(f"[{TAG}] done best_torch={best:.2f}", flush=True)
