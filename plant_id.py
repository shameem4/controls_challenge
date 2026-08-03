"""Full plant identification by PAIRED PERTURBATION along real trajectories.

Two previous system-ID efforts produced measurements that did not transfer to control, both for the
same reason: they were step tests from a HELD equilibrium (`sysid.py`). Holding 1-2 m/s^2 of lateral
acceleration constant for 150 steps is a state the closed loop never occupies -- real segments pass
through high lataccel, they do not sit there. The resulting recommendations (a left/right gain
correction, then a gain schedule on |lataccel|) both failed, the second in BOTH directions.

This measures the plant where it is actually used.

  1. drive real segments with a real controller
  2. at sampled step t, snapshot the full state AND the RNG
  3. roll forward HZ steps on the baseline action -> c_base
  4. restore exactly, roll forward HZ steps with action[t] + delta -> c_pert
  5. the difference c_pert - c_base IS the local impulse response at that state

Because the RNG is rewound, the noise realisation is identical in both arms and cancels exactly in
the difference -- common random numbers. That removes the drift that forced 96-sample averaging in
the bench tests, so every sample is informative and the variance is tiny.

Sampling is stratified by |lataccel| so hard cornering is represented far above its natural rate
(|tau| > 2 is 0.67% of steps but supplies 53% of the gap between the learned and classical
controllers, so it is the regime worth characterising).

Recorded per sample: the local response curve over HZ steps, plus every state variable available --
v_ego, roll, a_ego, current lataccel, recent action history, target and its local derivative. What
those variables mean physically is deliberately not assumed; the fit decides what matters.

Usage: python plant_id.py <nseg> [start] [tag]    env: DELTA, HZ, NSAMP, CTRL
"""
import sys, os, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, STEER_RANGE
from oracle_build import Stepper

DEV = 'cuda'
DELTA = float(os.environ.get('DELTA', 0.10))   # perturbation size, small enough to stay local
HZ = int(os.environ.get('HZ', 12))             # response window (impulse response spans ~6)
NSAMP = int(os.environ.get('NSAMP', 60))       # perturbation points per segment
CTRL = os.environ.get('CTRL', 'cnn_v2.pt')


@torch.no_grad()
def collect(plant, segs, net, seed=0):
    """Drive the segments and take paired perturbation measurements at stratified sample points."""
    B, T = len(segs), COST_END_IDX
    torch.manual_seed(seed)
    S = Stepper(plant, segs, T)
    pol = AblPolicy(net, B, DEV)

    # pass 1: drive normally, record |lataccel| so sampling can be stratified toward corners
    lat_trace = []
    while S.t < T:
        u = pol(S.ctx())
        lat_trace.append(S.cur.clone())
        S.step(u)
    lat_trace = torch.stack(lat_trace, 1).abs()          # [B, T-CONTEXT_LENGTH]

    # choose sample times: half uniform, half biased to the largest |lataccel| moments
    valid = np.arange(CONTROL_START_IDX, T - HZ - 1)
    per_half = NSAMP // 2
    times = []
    lt = lat_trace.cpu().numpy()
    off = valid[0] - 20
    for b in range(B):
        w = lt[b, valid - 20]
        hard = valid[np.argsort(-w)[:per_half * 3]]
        rng = np.random.default_rng(1000 + b)
        times.append(np.sort(np.concatenate([rng.choice(valid, per_half, replace=False),
                                             rng.choice(hard, per_half, replace=False)])))
    times = np.stack(times)                              # [B, NSAMP]

    # pass 2: replay, and at each sample time run the paired perturbation
    torch.manual_seed(seed)
    S = Stepper(plant, segs, T)
    pol = AblPolicy(net, B, DEV)
    ptr = np.zeros(B, dtype=int)
    FEAT, RESP = [], []
    hist_u = torch.zeros(B, T, device=DEV)
    while S.t < T:
        t = S.t
        ctx = S.ctx()
        u = pol(ctx)
        due = torch.tensor((ptr < NSAMP) & (times[np.arange(B), np.minimum(ptr, NSAMP - 1)] == t),
                           device=DEV)
        if due.any():
            snap = S.snapshot()
            # Measure the shift in the CONDITIONAL MEAN, not in a sampled trajectory. The plant
            # emits a discrete token, so with the RNG rewound both arms usually draw the SAME token
            # despite the shifted distribution -- giving exactly zero difference -- and occasionally
            # cross a bin boundary and jump. That produced median 0.000 with std 1.407. The state at
            # t is still the real sampled one, so the operating point stays realistic; only the
            # probe propagation is deterministic.
            S.mode = 'expected'
            pstate = (pol.integ.clone(), pol.prev.clone(), [a.clone() for a in pol.pact])
            # ARM 1: baseline. Record the actions taken so arm 2 can replay them exactly.
            base, acts = [], []
            uu = u
            for j in range(HZ):
                acts.append(uu)
                base.append(S.step(uu))
                uu = pol(S.ctx())
            S.restore(snap)
            pol.integ, pol.prev, pol.pact = pstate[0].clone(), pstate[1].clone(), [a.clone() for a in pstate[2]]
            # ARM 2: identical actions EXCEPT a one-step impulse at t. Letting the controller react
            # here instead would measure closed-loop disturbance rejection, which decays to zero by
            # construction -- an earlier version did exactly that and returned a median DC gain of
            # 0.000. Holding the actions fixed isolates the PLANT.
            pert = []
            for j in range(HZ):
                a_j = (acts[0] + DELTA).clamp(*STEER_RANGE) if j == 0 else acts[j]
                pert.append(S.step(a_j))
            S.restore(snap)
            pol.integ, pol.prev, pol.pact = pstate[0].clone(), pstate[1].clone(), [a.clone() for a in pstate[2]]
            S.mode = 'sample'                       # back to the true plant for the real trajectory
            # impulse response: only u[t] differs, so this is h[1..HZ] scaled by the local gain.
            # The RNG is rewound between arms, so the noise realisation is identical and cancels
            # exactly in the difference (common random numbers) -- no averaging needed.
            resp = (torch.stack(pert, 1) - torch.stack(base, 1)) / DELTA      # [B, HZ]
            fut = ctx['fut_lat']
            dtau = (fut[:, 0] - ctx['target']) if fut.shape[1] > 0 else torch.zeros(B, device=DEV)
            d2tau = (fut[:, 1] - 2 * fut[:, 0] + ctx['target']) if fut.shape[1] > 1 else torch.zeros(B, device=DEV)
            feat = torch.stack([ctx['v'], ctx['roll'], ctx['a'], ctx['cur'], ctx['target'],
                                ctx['target'] - ctx['cur'], u, hist_u[:, t - 1], hist_u[:, t - 2],
                                u - hist_u[:, t - 1], dtau, d2tau], -1)
            FEAT.append(torch.where(due[:, None], feat, torch.full_like(feat, float('nan'))).cpu())
            RESP.append(torch.where(due[:, None], resp, torch.full_like(resp, float('nan'))).cpu())
            ptr += due.cpu().numpy().astype(int)
        hist_u[:, t] = u
        S.step(u)
    F = torch.cat(FEAT, 0).numpy(); R = torch.cat(RESP, 0).numpy()
    ok = ~np.isnan(F[:, 0])
    return F[ok], R[ok]


FEAT_NAMES = ['v_ego', 'roll', 'a_ego', 'lataccel', 'target', 'error',
              'u', 'u[t-1]', 'u[t-2]', 'du', 'dtau', 'd2tau']


def main():
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    tag = sys.argv[3] if len(sys.argv) > 3 else 'pid'
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV); net.load_state_dict(torch.load(CTRL, map_location=DEV)); net.eval()
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    segs = [load_segment(f) for f in ALL[start:start + nseg]]
    print(f'[{tag}] segs={nseg} start={start} DELTA={DELTA} HZ={HZ} NSAMP={NSAMP} ctrl={CTRL}', flush=True)
    F, R = collect(plant, segs, net)
    print(f'[{tag}] collected {len(F)} paired perturbation samples', flush=True)
    np.savez(f'{tag}_{start}_{nseg}.npz', feat=F, resp=R, names=np.array(FEAT_NAMES))
    dc = R.sum(1) * 0 + R[:, -1]      # cumulative response by the end of the window
    a = np.abs(F[:, 3])
    print(f'  DC gain (response at HZ): mean {dc.mean():.3f}  median {np.median(dc):.3f}  '
          f'std {dc.std():.3f}', flush=True)
    print(f'  |lataccel| coverage: median {np.median(a):.3f}  p90 {np.percentile(a, 90):.3f}  '
          f'frac>1 {100*(a>1).mean():.1f}%  frac>2 {100*(a>2).mean():.1f}%', flush=True)


if __name__ == '__main__':
    main()
