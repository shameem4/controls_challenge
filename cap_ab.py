"""Is cnn model-limited? From-scratch A/B on network width.

Motivation. The released controller is AblNet('PM') with 11,243 parameters, against a plant that is
a ~1M-parameter transformer. The floor analysis (FINDINGS_TAIL.md §15) puts the achievable floor at
~29 while cnn's MEDIAN segment costs 44.33 -- so the shortfall is broad rather than tail-shaped, and
a broad shortfall from an 11k-parameter model points at capacity before anything else. Two standard
objections do not apply here: the reseeding test showed no memorisation, and the data is plentiful
(2000 segments x 400 steps), so scaling up carries little overfitting risk.

Everything already tried was a fine-tune of the converged small model (hard-example mining, uniform
fine-tuning, both null), which says nothing about whether the model itself is the limit. Hence a
from-scratch comparison: both arms get identical data, schedule, batch and iteration budget, and
differ ONLY in width. Comparing a new model against the released checkpoint would confound capacity
with a different training recipe, so the small arm is retrained here rather than reused.

Validation uses 200 segments rather than ablate.py's 60 -- this metric's subset noise is large
enough that 60 segments cannot resolve the differences at stake -- and reports MEDIAN alongside
mean, since a model that trades the bulk for the tail shows up in the median first.

GRADIENT ARM RERUN (2026-08-03). The ACC=12 arm was the only non-null cnn result (-1.269
[-2.26,-0.61]) but its own commit flagged three defects: arms matched on ITERATIONS not compute (so
3x data per iteration confounds data with gradient-variance reduction), never converged (best
checkpoint was the final iteration, still improving monotonically), and it loses to released cnn_PM
on ALL[:5000]. This rerun fixes the first two:

  * COMPUTE MATCHING -- set MATCH_ACC to the other arm's ACC and the iteration count is scaled by
    ACC_other/ACC_this, so both arms perform the same number of rollouts. Matching on iterations was
    the confound.
  * RESUME -- state (weights, optimiser, iteration, RNG) is written every SAVE_EVERY iterations and
    reloaded automatically, so a machine reboot costs at most SAVE_EVERY iterations rather than the
    run. This box rebooted twice on 2026-08-03 and silently killed two training runs.

Usage: python cap_ab.py <ch> <iters> [tag]
  env: ACC, MATCH_ACC, PREV_H, SAVE_EVERY, VAL_EVERY
"""
import sys, os, numpy as np, torch
from pathlib import Path
import nets
# Preview length must be set in TWO places: build_ff_window reads nets.H as a module global at call
# time, but AblNet's `h=H` default was bound at import, so it also has to be passed explicitly.
PREV_H = int(os.environ.get('PREV_H', nets.H))
nets.H = PREV_H
from torch_sim import Plant, load_segment, rollout, cost
from nets import AblNet, AblPolicy
from tinyphysics import COST_END_IDX

CH = int(sys.argv[1]) if len(sys.argv) > 1 else 32
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
TAG = sys.argv[3] if len(sys.argv) > 3 else f'ch{CH}'
CFG = 'PM'
DEV = 'cuda'
bs = 8
ACC = int(os.environ.get('ACC', 4))      # gradient-variance lever: effective batch = bs*ACC
# Compute matching: ITERS is expressed in units of the MATCH_ACC arm, then scaled so both arms run
# the same number of rollouts (ITERS*ACC). Without this the high-ACC arm simply sees 3x the data.
MATCH_ACC = int(os.environ.get('MATCH_ACC', 0))
if MATCH_ACC:
    ITERS = ITERS * MATCH_ACC // ACC
SAVE_EVERY = int(os.environ.get('SAVE_EVERY', 25))
VAL_EVERY = int(os.environ.get('VAL_EVERY', 25))
TBPTT = 30

ALL = sorted(Path('data/SYNTHETIC').iterdir())
TRAIN = ALL[2000:4000]
VAL = ALL[4000:4200]

plant = Plant(device=DEV)
net = AblNet(CFG, h=PREV_H, ch=CH, fb_hidden=CH, res_hidden=CH).to(DEV)
nparam = sum(p.numel() for p in net.parameters())
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
                tot.append(cost(traj, tg)[2].cpu().numpy())
    c = np.concatenate(tot)
    return float(c.mean()), float(np.median(c))


# ---- resume ----
STATE = f'ckpts/{TAG}_state.pt'
start_it = 0
if os.path.exists(STATE):
    # weights_only=False: the state holds numpy RNG state, which the (newer) default
    # weights_only=True refuses to unpickle. This file is written by this script only.
    st = torch.load(STATE, map_location=DEV, weights_only=False)
    net.load_state_dict(st['net']); opt.load_state_dict(st['opt'])
    start_it = st['it'] + 1
    np.random.set_state(st['np_rng']); torch.set_rng_state(st['torch_rng'].cpu())
    print(f"[{TAG}] RESUMED from iteration {start_it}", flush=True)

print(f"[{TAG}] cfg={CFG} ch={CH} params={nparam:,} iters={ITERS} "
      f"ACC={ACC} (eff batch {bs*ACC}) preview_H={PREV_H} "
      f"train={len(TRAIN)} val={len(VAL)}", flush=True)
best = (1e9, None)
for it in range(start_it, ITERS):
    stop = min(150 + it, COST_END_IDX)          # curriculum, as in ablate.py
    opt.zero_grad(); tl = 0.0
    for _ in range(ACC):
        idx = np.random.randint(0, len(TRAIN), bs)
        segs = [load_segment(TRAIN[k]) for k in idx]
        traj, tg = rollout(plant, segs, AblPolicy(net, bs, DEV), mode='gumbel',
                           tbptt=TBPTT, stop=stop, soft_tokens=True)
        loss = cost(traj, tg)[2].mean() / ACC
        loss.backward(); tl += loss.item()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if it % SAVE_EVERY == 0 or it == ITERS - 1:
        # Save EVERY periodic checkpoint, not just torch-val improvements: checkpoint selection is
        # done on the real numpy sim afterwards, and torch-val disagrees with it often enough that
        # discarding non-improving checkpoints would throw away the eventual winner.
        torch.save(net.state_dict(), f'ckpts/cap_{TAG}_{it:05d}.pt')
        torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'it': it,
                    'np_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state()}, STATE)
    if it % VAL_EVERY == 0 or it == ITERS - 1:
        mn, md = validate()
        tag = ''
        if mn < best[0]:
            best = (mn, f'ckpts/cap_{TAG}_{it:05d}.pt'); tag = ' *'
        print(f"[{TAG}] it{it:4d}/{ITERS} stop={stop} train={tl:6.2f} "
              f"val mean={mn:6.2f} median={md:6.2f}{tag}", flush=True)
print(f"[{TAG}] done params={nparam:,} best val mean={best[0]:.2f} -> {best[1]}", flush=True)
