"""Train the policy to track c*, the analytic cost-optimum, instead of minimising the raw cost.

Idea. The benchmark cost is `50*mean((lat-tau)^2)*100 + mean((dlat/dt)^2)*100` -- a quadratic penalty
on a DIFFERENCE, weighted 10000x. That makes a stiff, ill-conditioned landscape, and its gradients
travel through a stochastic plant, which is the same variance problem that wrecked the MPC inner
solver (see FINDINGS_NEURAL_MPC.md).

But the minimiser of that cost is known in closed form. For a perfect plant it is the Tikhonov /
Whittaker smoother of the target,

    (I + 2*D'D) c* = tau        with lambda = W_jerk/W_track = 2

already implemented and used by every classical controller here. So instead of asking the optimiser
to rediscover the tracking/jerk trade-off through a stiff objective, hand it the answer: regress the
plant's realised lataccel onto c*.

    loss = 5000 * mean((lat - c*)^2)          # no explicit jerk term at all

The jerk penalty is *implicit* -- c* is already smoothed by exactly the amount the cost function
wants. The hypothesis is that this is far better conditioned while asking for nearly the same thing.

Known mis-specification, stated up front: c* is optimal for a PERFECT plant. Ours has ~5 steps of
response lag, a rate clamp and random-walk noise, so c* is not exactly achievable -- which is
precisely why ff_pi needs a detuned feedforward on top of it. If pure c*-tracking underperforms, the
natural fallback is a blend (mostly c*, small residual jerk term) rather than abandoning the idea.

Validation always reports the BENCHMARK cost, never the training loss, so numbers stay comparable
to every other arm. Control: cap_ab.py from scratch, same width/pool/schedule/iterations
(val 48.62 / real sim 49.750).

Usage: python cstar_train.py <iters> [tag] [blend]
   blend=0 -> pure c* tracking;  blend>0 -> add blend * benchmark-jerk term
"""
import sys, os, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment, rollout, cost
from nets import AblNet, AblPolicy
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, DEL_T

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
TAG = sys.argv[2] if len(sys.argv) > 2 else 'cstar'
BLEND = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
CFG, DEV, CH = 'PM', 'cuda', 32
bs, ACC, TBPTT = 8, 4, 30
LAM = 2.0                                   # W_jerk / W_track -- the cost's own ratio

ALL = sorted(Path('data/SYNTHETIC').iterdir())
TRAIN = ALL[2000:4000]
VAL = ALL[4000:4200]


def tikhonov(tau, lam=LAM):
    """(I + lam*D'D) c = tau by the Thomas algorithm -- the analytic cost minimiser."""
    n = len(tau)
    d = np.full(n, 1 + 2 * lam); d[0] = d[-1] = 1 + lam
    a = np.full(n, -lam); c = np.full(n, -lam)
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = c[0] / d[0]; dp[0] = tau[0] / d[0]
    for i in range(1, n):
        den = d[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den if i < n - 1 else 0.0
        dp[i] = (tau[i] - a[i] * dp[i - 1]) / den
    x = np.zeros(n); x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


_CACHE = {}


def seg_and_cstar(f):
    """Segment plus its cost-optimal reference, cached -- the Thomas solve is a python loop and
    training touches ~13k segment-loads."""
    k = str(f)
    if k not in _CACHE:
        s = load_segment(f)
        _CACHE[k] = (s, tikhonov(s['target'][:COST_END_IDX].astype(np.float64)).astype(np.float32))
    return _CACHE[k]


def main():
    plant = Plant(device=DEV)
    net = AblNet(CFG, ch=CH, fb_hidden=CH, res_hidden=CH).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-4)
    os.makedirs('ckpts', exist_ok=True)


    def validate(seeds=(0, 1), chunk=40):
        tot = []
        for s in seeds:
            torch.manual_seed(s)
            with torch.no_grad():
                for i in range(0, len(VAL), chunk):
                    segs = [load_segment(f) for f in VAL[i:i + chunk]]
                    traj, tg = rollout(plant, segs, AblPolicy(net, len(segs), DEV),
                                       mode='sample', stop=COST_END_IDX)
                    tot.append(cost(traj, tg)[2].cpu().numpy())      # BENCHMARK cost, always
        c = np.concatenate(tot)
        return float(c.mean()), float(np.median(c))


    print(f"[{TAG}] c*-tracking objective (lambda={LAM}, blend={BLEND}) "
          f"iters={ITERS} | control from scratch: 48.62", flush=True)
    best = (1e9, None)
    for it in range(ITERS):
        stop = min(150 + it, COST_END_IDX)
        opt.zero_grad(); tl = 0.0
        for _ in range(ACC):
            idx = np.random.randint(0, len(TRAIN), bs)
            pairs = [seg_and_cstar(TRAIN[k]) for k in idx]
            segs = [p[0] for p in pairs]
            cst = torch.tensor(np.stack([p[1] for p in pairs]), device=DEV)
            traj, tg = rollout(plant, segs, AblPolicy(net, bs, DEV), mode='gumbel',
                               tbptt=TBPTT, stop=stop, soft_tokens=True)
            hi = min(stop, COST_END_IDX)
            c = traj[:, CONTROL_START_IDX:hi]
            ref = cst[:, CONTROL_START_IDX:hi]
            loss = ((c - ref) ** 2).mean() * 5000.0
            if BLEND > 0:
                loss = loss + BLEND * (((c[:, 1:] - c[:, :-1]) / DEL_T) ** 2).mean() * 100.0
            loss = loss / ACC
            loss.backward(); tl += loss.item()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if it % 25 == 0 or it == ITERS - 1:
            mn, md = validate()
            tag = ''
            if mn < best[0]:
                best = (mn, f'ckpts/{TAG}_{it:04d}.pt'); torch.save(net.state_dict(), best[1]); tag = ' *'
            print(f"[{TAG}] it{it:3d} stop={stop} trainloss={tl:7.2f} "
                  f"BENCHMARK val mean={mn:6.2f} median={md:6.2f}{tag}", flush=True)
    print(f"[{TAG}] done best benchmark val={best[0]:.2f} -> {best[1]}", flush=True)


if __name__ == '__main__':
    main()
