"""segment -> closed-form optimal trajectory c* -> ideal actions by RECEDING-HORIZON deconvolution.

Builds the teacher dataset for behaviour cloning: (observation, ideal_action) pairs, plus the
replayed cost so the teacher is verified before anything trains on it.

Three constructions were tried before this one and each failed for a distinct, recorded reason:

  one-step inversion   H[1]=0.02, so landing c*[t+1] needs ~50x amplification -> rails. Cost 116679.
  open-loop deconv     plans the whole sequence up front; +-9% gain error and every shock accumulate
                       with nothing correcting. Jerk was GOOD (18.26) but lataccel 347, total 365.3,
                       tracking error vs c* RMS 0.264. Only feedback rejects drift.
  action-sequence search  gradient diverges (chaos); greedy is myopic; greedy c*-tracking ignores the
                       jerk/noise tradeoff. See FINDINGS_ORACLE.md.

This version keeps the closed form but adds feedback. At step t the past actions are known and the
current lataccel is measured, so the plant's FREE response over the next Hh steps is fixed; solve

    min_u  || A u - (c*_future - roll - free) / G ||^2  +  mu || D u ||^2

for the next Hh actions, emit the first, advance, repeat. A and D never change, so the normal-matrix
factorisation is computed once. No search, no chaos: it is linear MPC on an analytic reference.

This is a legitimate teacher. It uses the plant MODEL and the known target trajectory, but never the
realised noise draws -- so unlike a seed-exploiting oracle, its skill is of a kind a causal student
could in principle inherit.

Usage: python ideal_mpc.py <nseg> [start] [tag]     env: HH, MU, LEAD
"""
import sys, os, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy, build_ff_window, build_multihorizon
from controllers.ff_pi import _thomas
from tinyphysics import (CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX, MAX_ACC_DELTA,
                         STEER_RANGE, DEL_T)
from oracle_build import Stepper

DEV = 'cuda'
HH = int(os.environ.get('HH', 20))            # control horizon
MU = float(os.environ.get('MU', 3e-3))        # smoothness regularisation on the action plan
LAM = 2.0                                     # = W_jerk / W_track, the cost-optimal Tikhonov weight
H_IMP = np.array([0.00, 0.02, 0.08, 0.22, 0.38, 0.30])
GAIN_FIT = np.load(Path(__file__).resolve().parent / 'gain_fit.npy')


def cstar(targets, lam=LAM):
    out = np.empty_like(targets)
    n = targets.shape[1]
    diag = np.full(n, 1 + 2 * lam); diag[0] = 1 + lam; diag[-1] = 1 + lam
    for b in range(targets.shape[0]):
        out[b] = _thomas(np.full(n, -lam), diag, np.full(n, -lam), targets[b].copy())
    return out


SSTEP = np.concatenate([np.cumsum(H_IMP), np.ones(200)])   # unit step response, saturating at 1


def build_solver(hh, mu):
    """Dynamic-matrix (DMC) solver, precomputed once -- it depends only on H.

    Predictions are anchored on the MEASURED current lataccel and model only the increment caused by
    deviating from the last applied action:

        c(t+1+j) ~= c(t) + G * sum_i SSTEP[j-i+1] * du_i ,   du_i = u(t+i) - u(t-1)

    Anchoring on c(t) is what makes this robust: accumulated drift and the +-9% gain error are
    already inside c(t), so they cancel instead of becoming a bias correction. An earlier version
    predicted ABSOLUTE lataccel from the 6-tap model and subtracted (model_now - c_now); that
    mismatch blew up and drove the command to the +-2 rails (mean|u_ideal-u_cnn| = 2.34).
    """
    A = np.zeros((hh, hh))
    for j in range(hh):
        for i in range(j + 1):
            A[j, i] = SSTEP[j - i + 1]
    # REACHABILITY WEIGHTING. SSTEP[1..4] = .02 .10 .32 .70, so the first few horizon steps are
    # almost unreachable: an unweighted least-squares tries to hit c*[t+1] anyway and over-drives,
    # which is what produced jerk 2388 and mean|du| 1.28. Weight each residual by how much the
    # command can actually move it.
    w = SSTEP[1:hh + 1].copy()
    A = A * w[:, None]
    D = np.zeros((hh - 1, hh))
    for i in range(hh - 1):
        D[i, i] = -1.0; D[i, i + 1] = 1.0                  # penalise action-rate
    return np.linalg.solve(A.T @ A + mu * (D.T @ D), A.T * w[None, :]), A


def main():
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    tag = sys.argv[3] if len(sys.argv) > 3 else 'imp'
    torch.manual_seed(0)
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV); net.load_state_dict(torch.load('cnn_v2.pt', map_location=DEV)); net.eval()
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    files = ALL[start:start + nseg]
    segs = [load_segment(f) for f in files]
    B, T = len(segs), COST_END_IDX

    tg = np.stack([s['target'][:T] for s in segs]).astype(np.float64)
    CS = cstar(tg)
    roll = np.stack([s['roll'][:T] for s in segs])
    vv = np.stack([s['v'][:T] for s in segs])
    G = np.clip(np.polyval(GAIN_FIT, vv), 0.3, 4.0)
    SOLVE, A = build_solver(HH, MU)
    print(f'[{tag}] segs={B} HH={HH} MU={MU}', flush=True)

    S = Stepper(plant, segs, T)
    base = AblPolicy(net, B, DEV)
    hist_u = np.zeros((B, T + HH))                 # actions actually applied, for the free response
    OBS, UI, UC, TRAJ, TGT = [], [], [], [], []

    while S.t < T:
        t = S.t
        ctx = S.ctx()
        u_cnn = base(ctx)
        if t < CONTROL_START_IDX:
            a = S.step(u_cnn)
            hist_u[:, t] = u_cnn.detach().cpu().numpy()
            continue

        hh = min(HH, T - t - 1)
        if hh < 1:
            break
        cur_np = S.cur.detach().cpu().numpy()
        uprev = hist_u[:, t - 1]
        # FREE RESPONSE. The increment model assumes the plant is settled with u held at u(t-1); it
        # is not -- past action CHANGES are still propagating through the 6-step impulse response.
        # Omitting this term is what produced jerk ~2300 and cost ~19000: the solver kept re-issuing
        # correction for motion that was already on its way.
        #   free_j = sum_{m>=1} (SSTEP[j+1+m] - SSTEP[m]) * du(t-m)
        free = np.zeros((B, hh))
        for m in range(1, len(H_IMP) + 1):
            if t - m - 1 < 0:
                break
            du_past = hist_u[:, t - m] - hist_u[:, t - m - 1]
            coef = SSTEP[np.arange(1, hh + 1) + m] - SSTEP[m]
            free += du_past[:, None] * coef[None, :]
        # desired INCREMENT from the measured current lataccel, in unit-gain plant units
        want = (CS[:, t + 1:t + 1 + hh] - cur_np[:, None]) / G[:, t + 1:t + 1 + hh] - free
        sol, _ = (SOLVE, A) if hh == HH else build_solver(hh, MU)
        du = (sol @ want.T).T if hh == HH else (sol @ want.T).T
        u_np = np.clip(uprev + du[:, 0], STEER_RANGE[0], STEER_RANGE[1])
        u = torch.tensor(u_np, dtype=torch.float32, device=DEV)

        e = ctx['target'] - ctx['cur']
        ffw = build_ff_window(ctx['target'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
        mh = build_multihorizon(ctx['cur'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
        OBS.append(torch.cat([torch.stack([e, base.integ, ctx['v'], ctx['roll'], ctx['cur']], -1),
                              ffw, mh], -1).detach().cpu())
        UI.append(u.cpu()); UC.append(u_cnn.detach().cpu()); TGT.append(ctx['target'].detach().cpu())
        TRAJ.append(S.step(u).detach().cpu())
        hist_u[:, t] = u_np
        if (t - CONTROL_START_IDX) % 100 == 0:
            print(f'  t={t}  mean|u_ideal-u_cnn|={(u - u_cnn).abs().mean().item():.4f}', flush=True)

    c = torch.stack(TRAJ, 1); tgt = torch.stack(TGT, 1)
    lat = ((c - tgt) ** 2).mean(1) * 5000.0
    jerk = (((c[:, 1:] - c[:, :-1]) / DEL_T) ** 2).mean(1) * 100.0
    print()
    print(f'[{tag}] TEACHER replay cost {(lat + jerk).mean().item():8.3f}  '
          f'(lataccel {lat.mean().item():.3f}  jerk {jerk.mean().item():.3f})', flush=True)
    print(f'[{tag}] reference: cnn_v2 ~43.6 on this split (torch, same seed)', flush=True)
    obs = torch.stack(OBS, 1)
    np.savez(f'{tag}_{start}_{nseg}.npz', obs=obs.numpy(), u_ideal=torch.stack(UI, 1).numpy(),
             u_cnn=torch.stack(UC, 1).numpy(), traj=c.numpy(), tgt=tgt.numpy(),
             files=np.array([str(f) for f in files]))
    print(f'[{tag}] saved obs {tuple(obs.shape)}', flush=True)


if __name__ == '__main__':
    main()
