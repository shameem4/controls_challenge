"""Stage 3 training: imitation warm-start (from ff_pi) + cost fine-tuning, two regimes.

  python train.py imitate                 -> cnn_warm.pt   (imitate ff_pi)
  python train.py finetune warm           -> cnn_A.pt      (warm-start -> cost)
  python train.py finetune scratch        -> cnn_B.pt      (random -> cost, curriculum)
  python train.py evaltorch <ckpt>        -> held-out sampled cost via torch engine
"""
import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch_sim import Plant, load_segment, rollout, cost
from nets import PreviewCNN, TorchPolicy, build_ff_window, make_net, H
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, FUTURE_PLAN_STEPS

DEV = 'cuda'
LAM = 25.06 / 12.5
GAIN_FIT = np.load('gain_fit.npy')
ALL = sorted(Path('data/SYNTHETIC').iterdir())
TRAIN = ALL[2000:2600]      # disjoint from tuning (first 40) and held-out (1000-1200)
VAL = ALL[1000:1120]


def smooth_full(tau, lam=LAM):
    B, T = tau.shape
    diag = torch.full((T,), 1 + 2 * lam, device=tau.device); diag[0] = diag[-1] = 1 + lam
    off = -lam
    cp = torch.zeros(T, device=tau.device); dp = torch.zeros(B, T, device=tau.device)
    cp[0] = off / diag[0]; dp[:, 0] = tau[:, 0] / diag[0]
    for i in range(1, T):
        m = diag[i] - off * cp[i - 1]
        cp[i] = (off / m) if i < T - 1 else 0.0
        dp[:, i] = (tau[:, i] - off * dp[:, i - 1]) / m
    x = torch.zeros(B, T, device=tau.device); x[:, -1] = dp[:, -1]
    for i in range(T - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[i] * x[:, i + 1]
    return x


def G(v, gain_scale=1.3):
    g = GAIN_FIT[0] * v * v + GAIN_FIT[1] * v + GAIN_FIT[2]
    return torch.clamp(gain_scale * g, 0.3, 4.0)


class Teacher:
    """ff_pi teacher; logs CNN-style features + its action each step."""
    def __init__(self, cstar, B, dev, kp=0.2, ki=0.1, lead=3, i_clip=5.0):
        self.cstar, self.kp, self.ki, self.lead, self.i_clip = cstar, kp, ki, lead, i_clip
        self.T = cstar.shape[1]
        self.integ = torch.zeros(B, device=dev)      # teacher PI (c*-based)
        self.cnn_integ = torch.zeros(B, device=dev)  # CNN-style (raw-target-based)
        self.cnn_prev = torch.zeros(B, device=dev)
        self.log = {'ff': [], 'v': [], 'fb': [], 'act': []}

    def __call__(self, ctx):
        t = ctx['step']
        desired = self.cstar[:, t]
        des_lead = self.cstar[:, min(t + self.lead, self.T - 1)]
        ff = (des_lead - ctx['roll']) / G(ctx['v'])
        e = desired - ctx['cur']
        self.integ = (self.integ + e).clamp(-self.i_clip, self.i_clip)
        act = ff + self.kp * e + self.ki * self.integ
        # log CNN features (raw-target based, matching TorchPolicy/eval)
        cnn_e = ctx['target'] - ctx['cur']
        self.cnn_integ = (self.cnn_integ + cnn_e).clamp(-self.i_clip, self.i_clip)
        ff_win = build_ff_window(ctx['target'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
        if t >= CONTROL_START_IDX:
            self.log['ff'].append(ff_win.detach())
            self.log['v'].append(ctx['v'].detach())
            self.log['fb'].append(torch.stack([cnn_e, self.cnn_integ, self.cnn_prev], -1).detach())
            self.log['act'].append(act.detach())
        self.cnn_prev = cnn_e
        return act


def batch_segs(files):
    segs = [load_segment(f) for f in files]
    T = min(len(s['target']) for s in segs)
    tau = torch.tensor(np.stack([s['target'][:T] for s in segs]), dtype=torch.float32, device=DEV)
    return segs, smooth_full(tau)


def gen_teacher_data(plant, files, bs=60):
    XF, XV, XB, YA = [], [], [], []
    for i in range(0, len(files), bs):
        fb = files[i:i + bs]
        segs, cstar = batch_segs(fb)
        teach = Teacher(cstar, len(segs), DEV)
        with torch.no_grad():
            rollout(plant, segs, teach, mode='sample')
        XF.append(torch.cat(teach.log['ff'])); XV.append(torch.cat(teach.log['v']))
        XB.append(torch.cat(teach.log['fb'])); YA.append(torch.cat(teach.log['act']))
        print(f"  teacher segs {i+len(fb)}/{len(files)}", flush=True)
    return (torch.cat(XF), torch.cat(XV), torch.cat(XB), torch.cat(YA))


def imitate():
    plant = Plant(device=DEV)
    print("generating teacher data...")
    XF, XV, XB, YA = gen_teacher_data(plant, TRAIN)
    print(f"dataset: {XF.shape[0]} samples")
    net = PreviewCNN().to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    N = XF.shape[0]; idx = torch.arange(N, device=DEV)
    for ep in range(15):
        perm = idx[torch.randperm(N, device=DEV)]
        tot = 0.0
        for j in range(0, N, 4096):
            b = perm[j:j + 4096]
            pred = net(XF[b], XV[b], XB[b])
            loss = ((pred - YA[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        print(f"  epoch {ep}: imitation MSE={tot/N:.5f}", flush=True)
    torch.save(net.state_dict(), 'cnn_warm.pt')
    print("saved cnn_warm.pt")


def cost_rollout(plant, net, files, mode='gumbel', bs=60):
    tot = []
    for i in range(0, len(files), bs):
        segs = [load_segment(f) for f in files[i:i + bs]]
        pol = TorchPolicy(net, len(segs), DEV)
        traj, target = rollout(plant, segs, pol, mode=mode, stop=COST_END_IDX)
        lat, jerk, t = cost(traj, target)
        tot.append(t)
    return torch.cat(tot)


def seeded_val(plant, net, files, seeds=(0, 1)):
    tots = []
    for s in seeds:
        torch.manual_seed(s)
        with torch.no_grad():
            tots.append(cost_rollout(plant, net, files, mode='sample').mean().item())
    return float(np.mean(tots))


def finetune(regime, iters=None):
    plant = Plant(device=DEV)
    net = PreviewCNN().to(DEV)
    ckpt = f'cnn_{"A" if regime == "warm" else "B"}.pt'
    if regime == 'warm':
        net.load_state_dict(torch.load('cnn_warm.pt')); iters = iters or 300; curric = False
    else:
        iters = iters or 500; curric = True         # horizon curriculum for cold start
    opt = torch.optim.Adam(net.parameters(), 2e-4)
    bs, ACC = 8, 4                                    # effective batch 32 via grad accumulation
    best = seeded_val(plant, net, VAL[:60])
    torch.save(net.state_dict(), ckpt)
    print(f"  init seeded_val={best:.2f}", flush=True)
    for it in range(iters):
        stop = min(150 + it, COST_END_IDX) if curric else COST_END_IDX
        opt.zero_grad(); tl = 0.0
        for _ in range(ACC):
            idx = torch.randint(len(TRAIN), (bs,))
            segs = [load_segment(TRAIN[k]) for k in idx]
            pol = TorchPolicy(net, bs, DEV)
            traj, target = rollout(plant, segs, pol, mode='gumbel', tbptt=30, stop=stop)
            loss = cost(traj, target)[2].mean() / ACC
            loss.backward(); tl += loss.item()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it % 25 == 0 or it == iters - 1:
            vc = seeded_val(plant, net, VAL[:60]); tag = ''
            if vc < best:
                best = vc; torch.save(net.state_dict(), ckpt); tag = ' *saved'
            print(f"  it {it:3d} stop={stop} train={tl:6.2f} seeded_val={vc:6.2f}{tag}", flush=True)
    print(f"best seeded_val={best:.2f}  saved {ckpt}")


def improve(tag, arch='base', ksamp=1, iters=250):
    """Scratch-curriculum fine-tune with K-sample gradient averaging; save periodic
    checkpoints to ckpts/ for later real-sim selection."""
    import os
    os.makedirs('ckpts', exist_ok=True)
    plant = Plant(device=DEV)
    net = make_net(arch).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-4)
    bs, ACC, K = 8, 4, ksamp
    best = 1e9
    print(f"improve tag={tag} arch={arch} ksamp={K} iters={iters}", flush=True)
    for it in range(iters):
        stop = min(150 + it, COST_END_IDX)              # horizon curriculum
        opt.zero_grad()
        tl = 0.0
        for _ in range(ACC):
            idx = torch.randint(len(TRAIN), (bs,))
            segs = [load_segment(TRAIN[k]) for k in idx]
            for _ in range(K):                          # K-sample gradient averaging
                pol = TorchPolicy(net, bs, DEV)
                traj, target = rollout(plant, segs, pol, mode='gumbel', tbptt=30, stop=stop)
                loss = cost(traj, target)[2].mean() / (ACC * K)
                loss.backward(); tl += loss.item()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it % 25 == 0 or it == iters - 1:
            torch.save(net.state_dict(), f'ckpts/{tag}_{it:04d}.pt')
            vc = seeded_val(plant, net, VAL[:60]); tag2 = ''
            if vc < best:
                best = vc; torch.save(net.state_dict(), f'ckpts/{tag}_besttorch.pt'); tag2 = ' *torchbest'
            print(f"  it {it:3d} stop={stop} train={tl:6.2f} torch_val={vc:6.2f}{tag2}", flush=True)
    print(f"done tag={tag}; best torch_val={best:.2f}; checkpoints in ckpts/{tag}_*.pt")


def evaltorch(ckpt):
    plant = Plant(device=DEV)
    net = PreviewCNN().to(DEV); net.load_state_dict(torch.load(ckpt)); net.eval()
    with torch.no_grad():
        c = cost_rollout(plant, net, VAL, mode='sample')
    print(f"{ckpt}: held-out sampled total = {c.mean().item():.2f}")


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'imitate': imitate()
    elif cmd == 'finetune': finetune(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif cmd == 'improve': improve(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == 'evaltorch': evaltorch(sys.argv[2])
