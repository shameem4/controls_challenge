"""Build a per-segment ORACLE action dataset by sequential greedy probing, and record the
observations a causal controller would have seen at each step.

Why not gradient descent. Optimizing an action sequence through this plant DIVERGES: warm-started
from cnn_v2's own actions (43.624), 50 Adam steps at lr 2e-4 with gradient clipping produced not one
iterate better than the warm start, sitting at 220-229 instead. The recursion is chaotic -- an action
change of ~0.005 flips sampled tokens and sends the trajectory somewhere unrelated -- so a 400-step
gradient is noise. This reproduces the project's earlier online-MPC failure (9353).

Why sequential greedy works instead. The past is never perturbed. At step t the history and the RNG
are fixed; we snapshot them, try K candidate actions, roll each forward a few steps to score it,
restore exactly, commit the best, and advance. Nothing diverges because nothing upstream changes.
This is derivative-free, so chaos is irrelevant -- and it is why leaderboard exploits describe
"online sim probing with RNG reset" rather than trajectory optimization.

Lookahead is HZ steps because the plant's impulse response spans ~6 steps
(H = [0, .02, .08, .22, .38, .30]); scoring only the next step would ignore most of an action's
effect. Beyond the probe window the base controller drives, so a candidate is judged on the
trajectory it actually leads to.

This is a TEACHER, not a controller. It uses privileged information -- it observes the realized
outcome of each candidate before choosing -- which no causal controller can do. Distilling it into a
net that reads only observations yields a legitimate controller; replaying its actions would be the
exploit. We are doing the former.

The known risk, stated up front: the oracle action is
    u_oracle = nominal(observable) + cancellation(realized draws, NOT observable)
and MSE distillation learns E[u_oracle | obs], which averages the second term toward zero because
the draws are white (measured |autocorr| <= 0.011). So the student may inherit only the nominal
part -- measured at 54.43 sampled, worse than cnn_v2's 46.26. `oracle_probe`-style analysis on this
dataset (how much of u_oracle is predictable from obs, and whether cnn_v2 already emits it) is the
gate before spending a training run.

Usage: python oracle_build.py <nseg> [start] [tag]     env: K, HZ, SPAN
"""
import sys, os, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy, build_ff_window, build_multihorizon
from tinyphysics import (CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX, FUTURE_PLAN_STEPS,
                         MAX_ACC_DELTA, STEER_RANGE, DEL_T)
from controllers.ff_pi import _thomas

DEV = 'cuda'
K = int(os.environ.get('K', 9))            # candidate actions per step
HZ = int(os.environ.get('HZ', 6))          # probe lookahead, ~ the impulse-response length
SPAN = float(os.environ.get('SPAN', 0.30))  # candidate spread in steering units
TRACK_CSTAR = os.environ.get('TRACK_CSTAR', '1') == '1'


class Stepper:
    """Batched plant stepper with exact snapshot/restore of history AND RNG.

    Only the last CONTEXT_LENGTH entries matter to the plant, so a snapshot is small. The RNG state
    must travel with it: rewinding the history but not the RNG would score candidates against
    different draws and silently defeat the whole point.
    """

    def __init__(self, plant, segs, T):
        self.p = plant
        self.T = T
        B = len(segs)
        st = lambda k: torch.tensor(np.stack([s[k][:T] for s in segs]), dtype=torch.float32, device=DEV)
        self.roll, self.v, self.a = st('roll'), st('v'), st('a')
        self.target, self.steer0 = st('target'), st('steer')
        self.lat = [self.target[:, i] for i in range(CONTEXT_LENGTH)]
        self.act = [self.steer0[:, i] for i in range(CONTEXT_LENGTH)]
        self.cur = self.lat[-1]
        self.t = CONTEXT_LENGTH
        self.mode = 'sample'      # probes may set 'expected' to measure the conditional mean

    def snapshot(self):
        return (list(self.lat[-CONTEXT_LENGTH:]), list(self.act[-CONTEXT_LENGTH:]),
                self.cur.clone(), self.t, torch.cuda.get_rng_state())

    def restore(self, s):
        self.lat, self.act, self.cur, self.t, rng = list(s[0]), list(s[1]), s[2].clone(), s[3], s[4]
        torch.cuda.set_rng_state(rng)

    def ctx(self):
        t = self.t
        e = min(t + FUTURE_PLAN_STEPS, self.T)
        return dict(target=self.target[:, t], cur=self.cur, roll=self.roll[:, t],
                    v=self.v[:, t], a=self.a[:, t], fut_lat=self.target[:, t + 1:e],
                    fut_roll=self.roll[:, t + 1:e], fut_v=self.v[:, t + 1:e], step=t)

    @torch.no_grad()
    def step(self, action):
        t = self.t
        action = action.clamp(*STEER_RANGE)
        if t < CONTROL_START_IDX:
            action = self.steer0[:, t]
        self.act.append(action)
        s = torch.stack([torch.stack(self.act[-CONTEXT_LENGTH:], 1),
                         torch.stack([self.roll[:, i] for i in range(t - CONTEXT_LENGTH + 1, t + 1)], 1),
                         torch.stack([self.v[:, i] for i in range(t - CONTEXT_LENGTH + 1, t + 1)], 1),
                         torch.stack([self.a[:, i] for i in range(t - CONTEXT_LENGTH + 1, t + 1)], 1)], -1)
        past = torch.stack(self.lat[-CONTEXT_LENGTH:], 1)
        pred = self.p.step(s, self.p.tokenize(past), mode=self.mode)
        pred = torch.clamp(pred, self.cur - MAX_ACC_DELTA, self.cur + MAX_ACC_DELTA)
        self.cur = torch.where(torch.tensor(t >= CONTROL_START_IDX, device=DEV), pred, self.target[:, t])
        self.lat.append(self.cur)
        self.t += 1
        return self.cur


def seg_cost(lats, tgts, prev_lat):
    """Benchmark cost of a probe window: 5000*mean(e^2) + 100*mean((dc/dt)^2).

    `prev_lat` (the lataccel BEFORE this window) is prepended for the jerk term only. Without it the
    transition INTO the window is uncharged, so each step picks its action independently, consecutive
    picks jump by up to the full candidate span, and the chatter shows up as jerk. That bug produced
    an "oracle" at 90.567 with jerk 49.0 -- twice cnn_v2's jerk and worse than the net it was meant
    to teach."""
    c = torch.stack(lats, 1); tg = torch.stack(tgts, 1)
    lat = ((c - tg) ** 2).mean(1) * 5000.0
    cj = torch.cat([prev_lat.unsqueeze(1), c], 1)
    jerk = (((cj[:, 1:] - cj[:, :-1]) / DEL_T) ** 2).mean(1) * 100.0
    return lat + jerk


def main():
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    tag = sys.argv[3] if len(sys.argv) > 3 else 'oracle'
    torch.manual_seed(0)
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV); net.load_state_dict(torch.load('cnn_v2.pt', map_location=DEV)); net.eval()
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    segs = [load_segment(f) for f in ALL[start:start + nseg]]
    B = len(segs); T = COST_END_IDX
    deltas = torch.linspace(-SPAN, SPAN, K, device=DEV)

    # Closed-form optimal lataccel trajectory. The benchmark cost is a convex quadratic in the
    # lataccel trajectory ALONE (independent of the plant), so its minimiser is the Tikhonov solve
    # (I + lam*D'D) c* = tau with lam = W_jerk/W_track = 2. Computed analytically at 6.69 here; an
    # independent implementation (RyanL2/commacontrol) reports 6.880 once the rate limit and the
    # 1024-bin quantisation are imposed. Tracking c* is a far better-posed oracle objective than
    # greedy window minimisation, which is myopic against a ~5-step impulse response and oscillates.
    lam = 2.0
    tg_np = np.stack([s_['target'][:T] for s_ in segs]).astype(np.float64)
    cstar = np.empty_like(tg_np)
    for b in range(len(segs)):
        n = T
        diag = np.full(n, 1 + 2 * lam); diag[0] = 1 + lam; diag[-1] = 1 + lam
        cstar[b] = _thomas(np.full(n, -lam), diag, np.full(n, -lam), tg_np[b].copy())
    CS = torch.tensor(cstar, dtype=torch.float32, device=DEV)

    S = Stepper(plant, segs, T)
    base = AblPolicy(net, B, DEV)
    OBS, ACT, TRAJ, TGT = [], [], [], []
    print(f'[{tag}] segs={B} K={K} HZ={HZ} SPAN={SPAN}', flush=True)

    while S.t < T:
        t = S.t
        ctx = S.ctx()
        with torch.no_grad():
            u0 = base(ctx)
        if t < CONTROL_START_IDX:
            S.step(u0)
            continue

        snap = S.snapshot()
        base_snap = (base.integ.clone(), base.prev.clone(), [a.clone() for a in base.pact])
        best_c = torch.full((B,), float('inf'), device=DEV)
        best_u = u0.clone()
        for d in deltas:
            S.restore(snap)
            base.integ, base.prev, base.pact = base_snap[0].clone(), base_snap[1].clone(), [a.clone() for a in base_snap[2]]
            cand = (u0 + d).clamp(*STEER_RANGE)
            prev_lat = S.cur.clone()
            lats, tgts = [], []
            u = cand
            for j in range(HZ):
                tgts.append(S.target[:, S.t])
                lats.append(S.step(u))
                if S.t >= T:
                    break
                with torch.no_grad():
                    u = base(S.ctx())
            if TRACK_CSTAR:
                # judge the candidate purely on how closely the plant lands on c*
                cc = torch.stack(lats, 1)
                tt = CS[:, t + 1:t + 1 + cc.shape[1]]
                m = min(cc.shape[1], tt.shape[1])          # window truncates at the segment end
                c = ((cc[:, :m] - tt[:, :m]) ** 2).mean(1)
            else:
                c = seg_cost(lats, tgts, prev_lat)
            better = c < best_c
            best_c = torch.where(better, c, best_c)
            best_u = torch.where(better, cand, best_u)

        # commit: rewind to t, record the observation a causal controller would have, apply best_u
        S.restore(snap)
        base.integ, base.prev, base.pact = base_snap[0].clone(), base_snap[1].clone(), [a.clone() for a in base_snap[2]]
        ctx = S.ctx()
        e = ctx['target'] - ctx['cur']
        ffw = build_ff_window(ctx['target'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
        mh = build_multihorizon(ctx['cur'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
        OBS.append(torch.cat([torch.stack([e, base.integ, ctx['v'], ctx['roll'], ctx['cur']], -1),
                              ffw, mh], -1).cpu())
        ACT.append(torch.stack([best_u, u0], -1).cpu())     # oracle action and cnn_v2's action
        with torch.no_grad():
            base(ctx)                                        # advance the base controller's state
        TGT.append(ctx['target'].cpu())
        TRAJ.append(S.step(best_u).cpu())
        if (t - CONTROL_START_IDX) % 50 == 0:
            print(f'  t={t}  mean|u_oracle - u_cnn| = {(best_u - u0).abs().mean().item():.4f}', flush=True)

    obs = torch.stack(OBS, 1); act = torch.stack(ACT, 1)
    c = torch.stack(TRAJ, 1); tg = torch.stack(TGT, 1)
    lat = ((c - tg) ** 2).mean(1) * 5000.0
    jerk = (((c[:, 1:] - c[:, :-1]) / DEL_T) ** 2).mean(1) * 100.0
    print(f'[{tag}] ORACLE cost {(lat + jerk).mean().item():.3f}  '
          f'(lataccel {lat.mean().item():.3f}  jerk {jerk.mean().item():.3f})', flush=True)
    np.savez(f'{tag}_{start}_{nseg}.npz', obs=obs.numpy(), act=act.numpy(),
             traj=c.numpy(), tgt=tg.numpy())
    print(f'[{tag}] saved obs {tuple(obs.shape)} act {tuple(act.shape)}', flush=True)


if __name__ == '__main__':
    main()
