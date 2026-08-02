"""Build oracle action sequences (privileged: optimized against a KNOWN noise realization) and ask
whether distilling them into a causal student can work at all.

Motivation. tinyphysics seeds each segment from md5(filename), so per segment the plant is a
deterministic function of the actions -- verified: open-loop replay reproduces closed-loop cost to
0.00e+00. That is what the sub-30 leaderboard entries exploit. But the same machinery can be used
HONESTLY: optimize actions against a known realization to get an expert, then distil that expert
into a controller that reads only observations. The student stays causal, so it is a legitimate
controller, not a lookup table.

We do NOT need the benchmark's specific seeds -- only demonstrations. So this runs entirely in the
differentiable torch sim with a fixed seed.

The reason to probe before training. The oracle action decomposes into
    u_oracle = nominal(observable state) + cancellation(future shocks, NOT observable)
Under MSE the student learns E[u_oracle | obs]. Shocks are white (measured: |autocorr| <= 0.011),
so the cancellation term averages to ZERO and the student converges toward the nominal-only policy
-- which was measured yesterday at 54.43 sampled, far worse than cnn_v2's 46.26.

So the questions that decide whether distillation has anything to offer:
  Q1  how much better is the oracle than cnn_v2? (sizes the prize, and confirms the exploit)
  Q2  what fraction of the oracle action is predictable from observations? (the student's ceiling)
  Q3  does cnn_v2 ALREADY emit that predictable part? If yes, distillation adds nothing.

Actions are optimized with Adam warm-started from cnn_v2's own actions. Starting from zero has
produced false "optimal" costs twice in this project -- worse than the controller being bounded --
so warm-starting is mandatory here.

Optimization runs through mode='gumbel' with soft_tokens, NOT mode='sample'. `sample` does
`self.bins[torch.multinomial(...)]` -- a hard index lookup with ZERO gradient -- so optimizing
through it would silently return the warm start unchanged. Gumbel-softmax with hard=True draws from
the same categorical distribution but is straight-through differentiable. With a fixed torch seed
the rollout is still a deterministic function of the actions, which is what the oracle requires.

Note on what "fixed noise" means here, because it is easy to state wrongly: what is fixed is the
STREAM OF RANDOM DRAWS, not an additive shock sequence. A draw maps to a lataccel token through the
CURRENT distribution, so changing the actions changes probs and the same draw yields a different
outcome. The disturbance is not separable from the trajectory. The exploit does not need it to be --
it needs only that action-sequence -> cost is deterministic, which it verifiably is.
"""
import sys, os, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment, rollout, cost
from nets import AblNet, AblPolicy, build_ff_window, build_multihorizon
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, STEER_RANGE


class FixedActions:
    """Replays a [B,T] action tensor; used to optimize actions directly through the plant."""
    def __init__(self, acts):
        self.acts = acts

    def detach_state(self):
        pass

    def __call__(self, ctx):
        return self.acts[:, ctx['step']]


def main():
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    dev = 'cuda'
    plant = Plant(device=dev)
    net = AblNet('PM').to(dev); net.load_state_dict(torch.load('cnn_v2.pt', map_location=dev)); net.eval()
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    segs = [load_segment(f) for f in ALL[6000:6000 + nseg]]
    torch.cuda.empty_cache()
    B = len(segs)

    # 1. cnn_v2 rollout: baseline cost, its actions (warm start), and the observations it saw
    torch.manual_seed(0)
    obs = []

    class Spy(AblPolicy):
        # subclass, not an instance patch: Python resolves __call__ on the CLASS, so assigning
        # pol.__call__ = fn is silently ignored (it was, and obs came back empty).
        def __call__(self, ctx):
            u = super().__call__(ctx)
            if ctx['step'] >= CONTROL_START_IDX:
                e = ctx['target'] - ctx['cur']
                ffw = build_ff_window(ctx['target'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
                mh = build_multihorizon(ctx['cur'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
                obs.append(torch.cat([torch.stack([e, self.integ, ctx['v'], ctx['roll'], ctx['cur']], -1),
                                      ffw, mh], -1).detach().cpu())
            return u
    pol = Spy(net, B, dev)
    with torch.no_grad():
        traj, target, acts0 = rollout(plant, segs, pol, mode='gumbel', stop=COST_END_IDX,
                                      soft_tokens=True, return_actions=True)
    base = cost(traj, target)[2]
    print(f'  cnn_v2 baseline (torch, fixed seed): {base.mean().item():.3f}', flush=True)

    # 2. optimize the action sequence against THIS realization -- the oracle
    A = acts0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([A], float(os.environ.get('ORACLE_LR', 2e-4)))
    bestA, bestc = A.detach().clone(), float('inf')
    for i in range(steps):
        opt.zero_grad()
        torch.manual_seed(0)
        traj, target = rollout(plant, segs, FixedActions(A.clamp(*STEER_RANGE)),
                               mode='gumbel', stop=COST_END_IDX, soft_tokens=True)
        c = cost(traj, target)[2].mean()
        c.backward()
        torch.nn.utils.clip_grad_norm_([A], 1.0)      # chaotic gradients: clip hard
        opt.step()
        if c.item() < bestc:
            bestc, bestA = c.item(), A.detach().clone()
        if (i + 1) % 25 == 0:
            print(f'    opt {i+1:4d}  cost {c.item():8.3f}   best {bestc:8.3f}', flush=True)
    A = bestA.requires_grad_(False)                    # keep the best iterate, never a worse one
    with torch.no_grad():
        torch.manual_seed(0)
        traj, target = rollout(plant, segs, FixedActions(A.clamp(*STEER_RANGE)),
                               mode='gumbel', stop=COST_END_IDX, soft_tokens=True)
        orc = cost(traj, target)[2]
    print(f'  ORACLE cost: {orc.mean().item():.3f}   (cnn_v2 {base.mean().item():.3f})', flush=True)

    # 3. how predictable is the oracle action from OBSERVATIONS?
    X = torch.stack(obs, 1).reshape(-1, obs[0].shape[-1]).numpy()          # [B*T, F]
    ua = A[:, CONTROL_START_IDX:COST_END_IDX].reshape(-1).cpu().numpy()
    uc = acts0.detach()[:, CONTROL_START_IDX:COST_END_IDX].reshape(-1).cpu().numpy()
    Z = np.column_stack([np.ones(len(X)), (X - X.mean(0)) / (X.std(0) + 1e-8)])

    def r2(y):
        b, *_ = np.linalg.lstsq(Z, y, rcond=None)
        p = Z @ b
        return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(), p

    r_or, p_or = r2(ua)
    r_cn, _ = r2(uc)
    print()
    print('=== Q2: is the oracle action learnable from observations? ===')
    print(f'  R^2(oracle action | obs)  = {r_or:.4f}')
    print(f'  R^2(cnn_v2 action | obs)  = {r_cn:.4f}   (sanity: it IS a function of obs)')
    print(f'  oracle action std {ua.std():.4f}   unexplained std {np.std(ua - p_or):.4f}')
    print()
    print('=== Q3: does cnn_v2 already emit the predictable part? ===')
    print(f'  corr(oracle, cnn_v2 actions)                 = {np.corrcoef(ua, uc)[0,1]:+.4f}')
    print(f'  corr(PREDICTABLE part of oracle, cnn_v2)     = {np.corrcoef(p_or, uc)[0,1]:+.4f}')
    print(f'  mean |oracle - cnn_v2|                       = {np.abs(ua - uc).mean():.4f}')
    print(f'  mean |predictable(oracle) - cnn_v2|          = {np.abs(p_or - uc).mean():.4f}')
    print(f'  std of the UNPREDICTABLE residual            = {np.std(ua - p_or):.4f}')
    np.savez('oracle_probe.npz', ua=ua, uc=uc, pred=p_or, base=base.cpu().numpy(), orc=orc.cpu().numpy())


if __name__ == '__main__':
    main()
