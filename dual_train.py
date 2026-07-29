"""Two-phase training: behaviour cloning first, then policy optimisation.

Hypothesis (worth stating precisely, because it is different from anything tested so far): policy
optimisation from scratch may converge to a particular basin, and starting it from an already-good
*and structurally different* region of weight space may reach a different one. Every previous
experiment varied the model (width), the optimiser (batch), or the inputs (preview) -- none varied
the INITIALISATION BASIN, and all of them converged to ~48.

This is the standard supervised-then-RL recipe (AlphaGo's policy network, and behaviour cloning
before RL fine-tuning generally), so the structure is well-founded rather than speculative.

  Phase 1  regress the net onto a TEACHER's actions with plain MSE. No plant backprop, no cost
           function -- ordinary supervised learning, so it is fast and lands wherever the teacher
           lives. Teacher choice matters: `ffpi` is a classical feedforward+PI controller, which
           occupies a genuinely different region than anything gradient-through-the-plant finds;
           `cnn` clones the existing policy into fresh weights.
  Phase 2  the usual policy optimisation -- Gumbel rollouts, soft-token BPTT, benchmark cost --
           starting from the phase-1 weights.

Behaviour cloning trains on the TEACHER's state distribution, so the student sees states it would
not itself visit; that is the standard distribution-shift problem. It needs no DAgger here because
phase 2 is on-policy and corrects it directly.

Control: cap_ab.py with the same width, pool, schedule and iteration count trained from scratch
(val 48.62 / real sim 49.750). Same phase-2 budget, so any difference is the initialisation.

Usage: python dual_train.py <ffpi|cnn> <bc_epochs> <po_iters> [tag]
"""
import sys, os, numpy as np, torch, torch.nn as nn
from pathlib import Path
import nets
from torch_sim import Plant, load_segment, rollout, cost
from nets import AblNet, AblPolicy, build_ff_window, build_multihorizon
from tinyphysics import COST_END_IDX

TEACHER = sys.argv[1] if len(sys.argv) > 1 else 'ffpi'
BC_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
PO_ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 400
TAG = sys.argv[4] if len(sys.argv) > 4 else f'dual_{TEACHER}'
CFG, DEV, CH = 'PM', 'cuda', 32
bs, ACC, TBPTT = 8, 4, 30
GF = np.load('gain_fit.npy')

ALL = sorted(Path('data/SYNTHETIC').iterdir())
TRAIN = ALL[2000:4000]
VAL = ALL[4000:4200]
BC_SEGS = ALL[2000:2400]          # subset of TRAIN, for cloning data

plant = Plant(device=DEV)
net = AblNet(CFG, ch=CH, fb_hidden=CH, res_hidden=CH).to(DEV)


def ffpi_teacher(B):
    """Classical inverse-plant feedforward + PI, ff_pi_tuned's gains. Structurally unlike anything
    policy optimisation converges to, which is the point of using it as a teacher."""
    st = {'integ': torch.zeros(B, device=DEV)}
    kp, ki, i_clip, scale, lead = 0.1424, 0.1353, 3.112, 1.79, 2

    def f(ctx):
        g = torch.zeros_like(ctx['v'])
        for c in GF:
            g = g * ctx['v'] + c
        g = (scale * g).clamp(0.3, 4.0)
        fl = ctx['fut_lat']
        ref = fl[:, lead] if fl.shape[1] > lead else ctx['target']
        ff = (ref - ctx['roll']) / g
        e = ctx['target'] - ctx['cur']
        st['integ'] = (st['integ'] + e).clamp(-i_clip, i_clip)
        return ff + kp * e + ki * st['integ']
    return f


TEACHER_CKPT = os.environ.get('TEACHER_CKPT', 'cnn_PM.pt')


def cnn_teacher(B):
    """Clone an existing policy. TEACHER_CKPT allows ITERATING the procedure: distil the previous
    round's output and optimise again, which tests whether behaviour cloning genuinely relocates the
    net to a more optimisable region (gains should repeat) or whether round 1 was a one-off escape
    from a sharp minimum (gains should vanish)."""
    t = AblNet(CFG).to(DEV)
    t.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEV))
    t.eval()
    for q in t.parameters():
        q.requires_grad_(False)
    return AblPolicy(t, B, DEV)


class Recorder(AblPolicy):
    """Builds exactly the features AblPolicy feeds the net, but ACTS with the teacher and records
    (features, teacher action). State advances on the teacher's actions so the recorded features
    stay consistent with the trajectory actually being flown."""
    def __init__(self, net, B, dev, teacher):
        super().__init__(net, B, dev)
        self.teacher = teacher
        self.buf = []

    def __call__(self, ctx):
        e = ctx['target'] - ctx['cur']
        self.integ = (self.integ + e).clamp(-self.i_clip, self.i_clip)
        ff_win = build_ff_window(ctx['target'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
        feats = [e, self.integ, self.prev] + self.pact[-self.net.n_prev:]
        fb = torch.stack(feats, -1)
        fb = torch.cat([fb, build_multihorizon(ctx['cur'], ctx['roll'],
                                               ctx['fut_lat'], ctx['fut_roll'], torch)], -1)
        u = self.teacher(ctx).detach()
        self.buf.append((ff_win.detach().cpu(), ctx['v'].detach().cpu(),
                         fb.detach().cpu(), u.cpu()))
        self.pact.append(u)
        self.prev = e
        return u


def validate(seeds=(0, 1), chunk=40):
    tot = []
    for s in seeds:
        torch.manual_seed(s)
        with torch.no_grad():
            for i in range(0, len(VAL), chunk):
                segs = [load_segment(f) for f in VAL[i:i + chunk]]
                traj, tg = rollout(plant, segs, AblPolicy(net, len(segs), DEV),
                                   mode='sample', stop=COST_END_IDX)
                tot.append(cost(traj, tg)[2].cpu().numpy())
    c = np.concatenate(tot)
    return float(c.mean()), float(np.median(c))


# ---------------- Phase 1: behaviour cloning ----------------
_tsrc = TEACHER if TEACHER == 'ffpi' else os.environ.get('TEACHER_CKPT', 'cnn_PM.pt')
print(f"[{TAG}] PHASE 1  teacher={_tsrc}  cloning on {len(BC_SEGS)} segments", flush=True)
X1, X2, X3, Y = [], [], [], []
for i in range(0, len(BC_SEGS), 40):
    segs = [load_segment(f) for f in BC_SEGS[i:i + 40]]
    tf = ffpi_teacher(len(segs)) if TEACHER == 'ffpi' else cnn_teacher(len(segs))
    rec = Recorder(net, len(segs), DEV, tf)
    torch.manual_seed(i)
    with torch.no_grad():
        rollout(plant, segs, rec, mode='sample', stop=COST_END_IDX)
    for a, b, c, d in rec.buf:
        X1.append(a); X2.append(b); X3.append(c); Y.append(d)
X1 = torch.cat(X1).to(DEV); X2 = torch.cat(X2).to(DEV)
X3 = torch.cat(X3).to(DEV); Y = torch.cat(Y).to(DEV)
print(f"[{TAG}] collected {len(Y):,} (state, action) pairs", flush=True)

opt = torch.optim.Adam(net.parameters(), 1e-3)
n = len(Y)
for ep in range(BC_EPOCHS):
    perm = torch.randperm(n, device=DEV)
    tot = 0.0
    for i in range(0, n, 4096):
        j = perm[i:i + 4096]
        opt.zero_grad()
        out = net(X1[j], X2[j], X3[j])
        loss = nn.functional.mse_loss(out, Y[j])
        loss.backward(); opt.step()
        tot += loss.item() * len(j)
    if ep % 5 == 0 or ep == BC_EPOCHS - 1:
        mn, md = validate()
        print(f"[{TAG}] bc ep{ep:3d} mse={tot/n:.5f}  closed-loop val mean={mn:7.2f} median={md:6.2f}",
              flush=True)
torch.save(net.state_dict(), f'ckpts/{TAG}_bc.pt')
del X1, X2, X3, Y
torch.cuda.empty_cache()

# ---------------- Phase 2: policy optimisation ----------------
print(f"\n[{TAG}] PHASE 2  policy optimisation, {PO_ITERS} iters (control from scratch: 48.62)",
      flush=True)
opt = torch.optim.Adam(net.parameters(), 2e-4)
best = (1e9, None)
for it in range(PO_ITERS):
    stop = min(150 + it, COST_END_IDX)
    opt.zero_grad(); tl = 0.0
    for _ in range(ACC):
        idx = np.random.randint(0, len(TRAIN), bs)
        segs = [load_segment(TRAIN[k]) for k in idx]
        traj, tg = rollout(plant, segs, AblPolicy(net, bs, DEV), mode='gumbel',
                           tbptt=TBPTT, stop=stop, soft_tokens=True)
        loss = cost(traj, tg)[2].mean() / ACC
        loss.backward(); tl += loss.item()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if it % 25 == 0 or it == PO_ITERS - 1:
        mn, md = validate()
        tag = ''
        if mn < best[0]:
            best = (mn, f'ckpts/{TAG}_po_{it:04d}.pt'); torch.save(net.state_dict(), best[1]); tag = ' *'
        print(f"[{TAG}] po it{it:3d} stop={stop} train={tl:6.2f} val mean={mn:6.2f} "
              f"median={md:6.2f}{tag}", flush=True)
print(f"[{TAG}] done best val mean={best[0]:.2f} -> {best[1]}", flush=True)
