"""segment -> optimal trajectory (closed form) -> ideal control actions (plant inversion) -> dataset.

No search over action sequences. Two exact steps:

1. THE OPTIMAL TRAJECTORY IS CLOSED FORM. Over the scored window the benchmark cost
   `5000*mean((c-tau)^2) + 100*mean((dc/dt)^2)` is a convex quadratic in the lataccel trajectory `c`
   alone -- it does not involve the plant. Its minimiser is the Tikhonov / Whittaker solve
       (I + lam * D'D) c* = tau ,   lam = W_jerk / W_track = 2
   solved in O(n) by the Thomas algorithm. We compute its cost at 6.69 (lataccel 1.53 + jerk 5.16);
   RyanL2/commacontrol reports 6.880 for the same quantity once the rate limit and 1024-bin
   quantisation are imposed. Two independent derivations.

2. IDEAL ACTIONS COME FROM INVERTING THE PLANT, NOT FROM OPTIMISING. At step t we know where we want
   to land: c*[t+1]. The plant's expected next lataccel is monotone in the steer command, so a
   bisection on u in STEER_RANGE finds the action that lands exactly there. ~12 batched plant calls
   per step, deterministic, no chaos.

Why this is not the oracle-search that failed. Three earlier attempts optimised ACTION SEQUENCES --
gradient (diverged, chaos), greedy window (myopic), greedy c*-tracking without a jerk term (jerk
131). Here nothing is searched: the trajectory is analytic and each action is a scalar root-find.

Honesty note. Inversion is done against the plant's EXPECTED next lataccel, then the true sampled
plant is stepped. So the actions are "ideal" in the certainty-equivalent sense, computed with full
knowledge of the target trajectory but NOT of the realised draws. That makes this a legitimate
teacher whose skill a causal student could in principle inherit -- unlike a seed-exploiting oracle,
whose advantage is a function of draws the student can never see.

Expect the replay cost to land well above 6.69: c* is optimal as a trajectory you could impose
directly, but the plant has ~5-step dead time and the noise pushes you off it every step. Measuring
that gap is the point of the verification pass.

Usage: python ideal_actions.py <nseg> [start] [tag]      env: BISECT
"""
import sys, os, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy, build_ff_window, build_multihorizon
from controllers.ff_pi import _thomas
from tinyphysics import (CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX, FUTURE_PLAN_STEPS,
                         MAX_ACC_DELTA, STEER_RANGE, DEL_T)
from oracle_build import Stepper

DEV = 'cuda'
MU = float(os.environ.get('MU', 1e-3))          # deconvolution regularisation
LAM = 2.0


def cstar(targets, lam=LAM):
    """Closed-form cost-optimal lataccel trajectory for each segment. O(n) tridiagonal solve."""
    out = np.empty_like(targets)
    n = targets.shape[1]
    diag = np.full(n, 1 + 2 * lam); diag[0] = 1 + lam; diag[-1] = 1 + lam
    for b in range(targets.shape[0]):
        out[b] = _thomas(np.full(n, -lam), diag, np.full(n, -lam), targets[b].copy())
    return out


@torch.no_grad()
def expected_next(S, u):
    """Plant's EXPECTED next lataccel for action u, without advancing the stepper."""
    t = S.t
    act = S.act[-CONTEXT_LENGTH + 1:] + [u.clamp(*STEER_RANGE)]
    s = torch.stack([torch.stack(act, 1),
                     torch.stack([S.roll[:, i] for i in range(t - CONTEXT_LENGTH + 1, t + 1)], 1),
                     torch.stack([S.v[:, i] for i in range(t - CONTEXT_LENGTH + 1, t + 1)], 1),
                     torch.stack([S.a[:, i] for i in range(t - CONTEXT_LENGTH + 1, t + 1)], 1)], -1)
    past = torch.stack(S.lat[-CONTEXT_LENGTH:], 1)
    pred = S.p.step(s, S.p.tokenize(past), mode='expected')
    return torch.clamp(pred, S.cur - MAX_ACC_DELTA, S.cur + MAX_ACC_DELTA)


H_IMP = np.array([0.00, 0.02, 0.08, 0.22, 0.38, 0.30])     # measured impulse response, unit DC gain
GAIN_FIT = np.load(Path(__file__).resolve().parent / 'gain_fit.npy')


def deconvolve(cs, roll, v, mu):
    """Ridge deconvolution: find the action sequence whose plant response tracks c*.

    ONE-STEP INVERSION DOES NOT WORK on this plant. H[1] = 0.02, so an action moves lataccel by only
    2% of its DC gain one step later; solving `E[c(t+1)] = c*[t+1]` for u(t) demands a ~50x
    amplification and drives the command straight to the +-2 rails (measured: mean|u_ideal-u_cnn|
    = 2.57, replay cost 116679). Dead time means the action that produces a given trajectory is
    spread over ~6 steps, so the inverse is a DECONVOLUTION over the sequence, not a per-step solve.

        c = roll + G * (H * u)      ->      minimise ||H*u - (c*-roll)/G||^2 + mu*||D u||^2

    H is strongly low-pass, so exact deconvolution is ill-conditioned and would produce huge
    oscillating commands; the second-difference penalty `mu*||D u||^2` regularises it. Closed form,
    no search -- one banded least-squares solve per segment.
    """
    n = len(cs)
    G = np.clip(np.polyval(GAIN_FIT, v), 0.3, 4.0)
    y = (cs - roll) / G                                     # desired unit-gain plant response
    A = np.zeros((n, n))
    for k, h in enumerate(H_IMP):
        if h:
            idx = np.arange(k, n)
            A[idx, idx - k] = h
    D = np.zeros((n - 2, n))                                # second difference: penalise curvature
    for i in range(n - 2):
        D[i, i] = 1.0; D[i, i + 1] = -2.0; D[i, i + 2] = 1.0
    M = A.T @ A + mu * (D.T @ D)
    return np.linalg.solve(M, A.T @ y)


def main():
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    tag = sys.argv[3] if len(sys.argv) > 3 else 'ideal'
    torch.manual_seed(0)
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV); net.load_state_dict(torch.load('cnn_v2.pt', map_location=DEV)); net.eval()
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    files = ALL[start:start + nseg]
    segs = [load_segment(f) for f in files]
    B, T = len(segs), COST_END_IDX

    tg = np.stack([s['target'][:T] for s in segs]).astype(np.float64)
    CS = torch.tensor(cstar(tg), dtype=torch.float32, device=DEV)
    print(f'[{tag}] segs={B} mu={MU}', flush=True)

    roll_np = np.stack([s_['roll'][:T] for s_ in segs])
    v_np = np.stack([s_['v'][:T] for s_ in segs])
    CS_np = cstar(tg)
    U_PLAN = np.stack([deconvolve(CS_np[b], roll_np[b], v_np[b], MU) for b in range(B)])
    U_PLAN = np.clip(U_PLAN, STEER_RANGE[0], STEER_RANGE[1])
    UP = torch.tensor(U_PLAN, dtype=torch.float32, device=DEV)
    print(f'[{tag}] deconvolved actions: mean|u| {np.abs(U_PLAN).mean():.3f}  '
          f'max|u| {np.abs(U_PLAN).max():.3f}  frac at rail {np.mean(np.abs(U_PLAN) > 1.99):.4f}', flush=True)

    S = Stepper(plant, segs, T)
    base = AblPolicy(net, B, DEV)                      # only to record what cnn_v2 would have done
    OBS, U_IDEAL, U_CNN, TRAJ, TGT = [], [], [], [], []
    while S.t < T:
        t = S.t
        ctx = S.ctx()
        u_cnn = base(ctx)
        if t < CONTROL_START_IDX:
            S.step(u_cnn)
            continue
        u = UP[:, t]

        e = ctx['target'] - ctx['cur']
        ffw = build_ff_window(ctx['target'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
        mh = build_multihorizon(ctx['cur'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
        OBS.append(torch.cat([torch.stack([e, base.integ, ctx['v'], ctx['roll'], ctx['cur']], -1),
                              ffw, mh], -1).cpu())
        U_IDEAL.append(u.cpu()); U_CNN.append(u_cnn.cpu())
        TGT.append(ctx['target'].cpu())
        TRAJ.append(S.step(u).cpu())
        if (t - CONTROL_START_IDX) % 100 == 0:
            print(f'  t={t}  mean|u_ideal-u_cnn|={(u - u_cnn).abs().mean().item():.4f}', flush=True)

    c = torch.stack(TRAJ, 1); tgt = torch.stack(TGT, 1)
    lat = ((c - tgt) ** 2).mean(1) * 5000.0
    jerk = (((c[:, 1:] - c[:, :-1]) / DEL_T) ** 2).mean(1) * 100.0
    cs_t = CS[:, CONTROL_START_IDX:T].cpu()
    print()
    print(f'[{tag}] IDEAL-ACTION replay cost {(lat + jerk).mean().item():8.3f}  '
          f'(lataccel {lat.mean().item():.3f}  jerk {jerk.mean().item():.3f})', flush=True)
    print(f'[{tag}] tracking error vs c*: RMS {((c - cs_t) ** 2).mean().sqrt().item():.4f}', flush=True)
    obs = torch.stack(OBS, 1); ui = torch.stack(U_IDEAL, 1); uc = torch.stack(U_CNN, 1)
    np.savez(f'{tag}_{start}_{nseg}.npz', obs=obs.detach().numpy(), u_ideal=ui.detach().numpy(),
             u_cnn=uc.detach().numpy(), traj=c.detach().numpy(), tgt=tgt.detach().numpy(),
             files=np.array([str(f) for f in files]))
    print(f'[{tag}] saved obs {tuple(obs.shape)}  u_ideal {tuple(ui.shape)}', flush=True)


if __name__ == '__main__':
    main()
