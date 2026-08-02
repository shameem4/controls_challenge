"""Behaviour-clone the per-segment `steer_lookup` teacher into a causal CNN.

Teacher: coordinate-descent-optimised steering per segment, tuned against that segment's fixed noise
realisation (steer_lookup.py). It beats cnn_v2 on the segments it was built for -- 37.819 vs 43.624
on the 16-segment probe -- which is what makes distillation worth attempting at all.

Student: the same AblNet architecture cnn_v2 uses, reading ONLY observations. It never sees which
segment it is in and never sees the realised draws, so it stays a legitimate controller.

Training follows the segment-by-segment structure: one segment at a time, timesteps in order, MSE
against the teacher's action at each step. Segment order is shuffled each epoch.

Reconstructing AblNet's inputs. The dataset stores [e, integ, v_ego, roll, cur] + ff_window(26) +
multihorizon(6). AblNet's feedback vector is [e, integ, prev_e, pact(3)] + multihorizon, so:
  prev_e  = e at t-1 within the segment (0 at t=0)
  pact    = the three previous TEACHER actions, since those are what was actually applied
Both are recoverable from the stored sequence, which is why the dataset keeps time order.

The known ceiling, stated up front so the result is interpretable either way. The teacher's action is
    u_teacher = nominal(observable) + cancellation(realised draws, NOT observable)
and MSE regression converges to E[u_teacher | obs]. The draws are white (|autocorr| <= 0.011), so the
second term averages toward zero and the student can only inherit the nominal part. The gap between
teacher (37.8) and student measures exactly how much of the teacher's edge was privileged.

Usage: python bc_train.py <npz> [epochs] [tag]      env: SEED, LR
"""
import sys, os, glob, numpy as np, torch, torch.nn as nn
from pathlib import Path
from nets import AblNet, N_PREV

DEV = 'cuda'


def load(paths):
    OBS, UI = [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        OBS.append(d['obs']); UI.append(d['u_ideal'])
    return np.concatenate(OBS, 0), np.concatenate(UI, 0)


def to_inputs(obs, ui):
    """[B,T,37] + [B,T] -> ff_win [B,T,26], v [B,T], fb [B,T,12] in AblNet's expected layout."""
    B, T, _ = obs.shape
    e, integ, v = obs[..., 0], obs[..., 1], obs[..., 2]
    ffw, mh = obs[..., 5:31], obs[..., 31:37]
    prev = np.concatenate([np.zeros((B, 1), np.float32), e[:, :-1]], 1)
    pact = np.zeros((B, T, N_PREV), np.float32)
    for k in range(1, N_PREV + 1):
        pact[:, k:, N_PREV - k] = ui[:, :-k]          # the teacher's own previous actions
    fb = np.concatenate([e[..., None], integ[..., None], prev[..., None], pact, mh], -1)
    return ffw.astype(np.float32), v.astype(np.float32), fb.astype(np.float32)


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else 'sl*_*.npz'
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    tag = sys.argv[3] if len(sys.argv) > 3 else 'bc'
    seed = int(os.environ.get('SEED', 0))
    lr = float(os.environ.get('LR', 1e-3))
    torch.manual_seed(seed); np.random.seed(seed)

    paths = sorted(glob.glob(pat))
    obs, ui = load(paths)
    ffw, v, fb = to_inputs(obs, ui)
    B, T = ui.shape
    print(f'[{tag}] files={len(paths)} segments={B} steps={T} samples={B*T}', flush=True)

    ffw_t = torch.tensor(ffw, device=DEV); v_t = torch.tensor(v, device=DEV)
    fb_t = torch.tensor(fb, device=DEV); y_t = torch.tensor(ui, dtype=torch.float32, device=DEV)

    net = AblNet('PM').to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr)
    nval = max(1, B // 8)
    perm = np.random.permutation(B)
    val_idx, tr_idx = perm[:nval], perm[nval:]
    print(f'[{tag}] train {len(tr_idx)} segs, val {len(val_idx)} segs', flush=True)

    best = float('inf')
    for ep in range(epochs):
        net.train()
        order = np.random.permutation(tr_idx)
        tot = 0.0
        for b in order:                                  # one SEGMENT at a time, timesteps in order
            opt.zero_grad()
            pred = net(ffw_t[b], v_t[b], fb_t[b])        # [T]
            loss = ((pred - y_t[b]) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        net.eval()
        with torch.no_grad():
            vl = np.mean([((net(ffw_t[b], v_t[b], fb_t[b]) - y_t[b]) ** 2).mean().item()
                          for b in val_idx])
        mark = ''
        if vl < best:
            best, mark = vl, ' *'
            torch.save(net.state_dict(), f'ckpts/{tag}_best.pt')
        if ep % 5 == 0 or ep == epochs - 1:
            print(f'[{tag}] ep{ep:3d} train_mse={tot/len(order):.6f} val_mse={vl:.6f}{mark}', flush=True)
    print(f'[{tag}] done best_val_mse={best:.6f} -> ckpts/{tag}_best.pt', flush=True)

    # how much of the teacher is even predictable from observations? This is the ceiling on any
    # student, and it is the number that explains wherever the student lands.
    net.eval()
    with torch.no_grad():
        p = torch.cat([net(ffw_t[b], v_t[b], fb_t[b]) for b in val_idx]).cpu().numpy()
    yv = y_t[val_idx].reshape(-1).cpu().numpy()
    uc = np.concatenate([np.load(p_, allow_pickle=True)['u_cnn'] for p_ in paths], 0)[val_idx].reshape(-1)
    ss = 1 - ((yv - p) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    ss_cnn = 1 - ((yv - uc) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    print(f'[{tag}] R^2(teacher | student prediction) = {ss:.4f}', flush=True)
    print(f'[{tag}] R^2(teacher | cnn_v2 action)      = {ss_cnn:.4f}   '
          f'(how much cnn_v2 ALREADY matches the teacher)', flush=True)


if __name__ == '__main__':
    main()
