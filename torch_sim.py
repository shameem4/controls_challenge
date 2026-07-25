"""Stage 2b: differentiable TinyPhysics rollout in PyTorch (batched, GPU).

Plant.step supports:
  mode='expected' : E[lataccel] = sum(softmax(logits/T) * bins)   (deterministic, differentiable in states)
  mode='sample'   : hard multinomial sample (true stochastic recursion, non-diff)
  mode='gumbel'   : straight-through Gumbel-softmax sample (stochastic AND differentiable)

Token feedback (past lataccels -> tokens -> embedding) is discrete. For gradients we
detach tokens by default (myopic / 1-step-truncated BPTT); soft-token full BPTT is a later upgrade.
"""
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from onnx2torch import convert
from tinyphysics import (CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX, DEL_T,
    LAT_ACCEL_COST_MULTIPLIER, LATACCEL_RANGE, VOCAB_SIZE, MAX_ACC_DELTA, ACC_G,
    FUTURE_PLAN_STEPS, STEER_RANGE)

def make_bins(device):
    return torch.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE, device=device)


class _SoftGather(torch.nn.Module):
    """Replaces the token-embedding Gather so it also accepts a soft one-hot [B,20,VOCAB]:
    int indices -> normal lookup; float weights -> weights @ embedding. Enables gradients
    through the plant's autoregressive lataccel feedback (soft-token BPTT)."""
    def __init__(self, orig): super().__init__(); self.orig = orig
    def forward(self, data, indices):
        if indices.dtype in (torch.long, torch.int64, torch.int32):
            return self.orig(data, indices)
        return indices @ data                       # [B,20,VOCAB] @ [VOCAB,64] -> [B,20,64]


class Plant:
    def __init__(self, onnx='models/tinyphysics.onnx', device='cuda', soft_tau=0.005):
        self.m = convert(onnx).to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.m._modules['wt2_embedding/Gather'] = _SoftGather(self.m._modules['wt2_embedding/Gather'])
        self.bins = make_bins(device)
        self.device = device
        self.soft_tau = soft_tau

    def tokenize(self, x):
        # match np.digitize(clip(x), bins, right=True)
        x = x.clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return torch.searchsorted(self.bins, x, right=False).clamp(0, VOCAB_SIZE - 1)

    def soft_tokens(self, past):
        """Straight-through soft one-hot over vocab for past lataccels [B,20]:
        exact hard forward, differentiable (Gaussian-soft) backward."""
        d = -(past.unsqueeze(-1) - self.bins).pow(2) / self.soft_tau
        w_soft = F.softmax(d, -1)
        w_hard = F.one_hot(self.tokenize(past), VOCAB_SIZE).float()
        return w_hard + (w_soft - w_soft.detach())

    def step(self, states, tokens, temperature=0.8, mode='expected'):
        # states [B,20,4] float, tokens [B,20] long
        logits = self.m(states, tokens)[:, -1]  # [B,1024]
        if mode == 'expected':
            probs = F.softmax(logits / temperature, -1)
            return (probs * self.bins).sum(-1)
        if mode == 'sample':
            probs = F.softmax(logits / temperature, -1)
            idx = torch.multinomial(probs, 1).squeeze(-1)
            return self.bins[idx]
        if mode == 'gumbel':
            oh = F.gumbel_softmax(logits / temperature, tau=1.0, hard=True)  # ST one-hot
            return (oh * self.bins).sum(-1)
        raise ValueError(mode)


def load_segment(path):
    df = pd.read_csv(path)
    return dict(
        roll=np.sin(df['roll'].values) * ACC_G,
        v=df['vEgo'].values, a=df['aEgo'].values,
        target=df['targetLateralAcceleration'].values,
        steer=-df['steerCommand'].values,
    )


def rollout(plant, segs, controller, mode='expected', detach_tokens=True,
            tbptt=None, stop=None, return_actions=False, soft_tokens=False):
    """Batched rollout mirroring TinyPhysicsSimulator. `controller(ctx)->steer[B]`.
    tbptt: if set, detach recurrent state every `tbptt` steps (bounds autograd memory).
    stop: if set, end the rollout at this step (e.g. COST_END_IDX for training)."""
    dev = plant.device
    T = min(len(s['target']) for s in segs)
    if stop is not None:
        T = min(T, stop)
    B = len(segs)
    def stk(k):
        return torch.tensor(np.stack([s[k][:T] for s in segs]), dtype=torch.float32, device=dev)
    roll, v, a, target, steer0 = stk('roll'), stk('v'), stk('a'), stk('target'), stk('steer')

    # history buffers as lists of [B] tensors
    lat_hist = [target[:, i] for i in range(CONTEXT_LENGTH)]
    act_hist = [steer0[:, i] for i in range(CONTEXT_LENGTH)]
    state_roll = [roll[:, i] for i in range(CONTEXT_LENGTH)]
    state_v = [v[:, i] for i in range(CONTEXT_LENGTH)]
    state_a = [a[:, i] for i in range(CONTEXT_LENGTH)]
    cur = lat_hist[-1]

    lat_traj = list(lat_hist)
    for t in range(CONTEXT_LENGTH, T):
        # controller
        fp_end = min(t + FUTURE_PLAN_STEPS, T)
        ctx = dict(target=target[:, t], cur=cur, roll=roll[:, t], v=v[:, t], a=a[:, t],
                   fut_lat=target[:, t + 1:fp_end], fut_roll=roll[:, t + 1:fp_end],
                   fut_v=v[:, t + 1:fp_end], step=t,
                   # extra context for model-based controllers (MPC): the *applied* action and
                   # lataccel history windows, plus the full exogenous arrays to slice from.
                   act_win=act_hist[-CONTEXT_LENGTH:], lat_win=lat_hist[-CONTEXT_LENGTH:],
                   roll_all=roll, v_all=v, a_all=a, T=T)
        act = controller(ctx)
        if t < CONTROL_START_IDX:
            act = steer0[:, t]
        act = act.clamp(STEER_RANGE[0], STEER_RANGE[1])
        act_hist.append(act)
        state_roll.append(roll[:, t]); state_v.append(v[:, t]); state_a.append(a[:, t])

        # assemble model inputs from last 20
        st = torch.stack([
            torch.stack(act_hist[-CONTEXT_LENGTH:], 1),
            torch.stack(state_roll[-CONTEXT_LENGTH:], 1),
            torch.stack(state_v[-CONTEXT_LENGTH:], 1),
            torch.stack(state_a[-CONTEXT_LENGTH:], 1)], dim=-1)  # [B,20,4]
        past = torch.stack(lat_hist[-CONTEXT_LENGTH:], 1)         # [B,20] values
        if soft_tokens:
            tok = plant.soft_tokens(past)                        # differentiable feedback (full BPTT)
        else:
            tok = plant.tokenize(past.detach() if detach_tokens else past)
        pred = plant.step(st, tok, mode=mode)
        pred = torch.clamp(pred, cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA)
        cur = torch.where(torch.tensor(t >= CONTROL_START_IDX, device=dev), pred, target[:, t])
        if tbptt is not None and (t - CONTEXT_LENGTH) % tbptt == 0:
            cur = cur.detach()
            act_hist[-1] = act_hist[-1].detach()
            if hasattr(controller, 'detach_state'):
                controller.detach_state()
        lat_hist.append(cur)
        lat_traj.append(cur)

    traj = torch.stack(lat_traj, 1)  # [B,T]
    if return_actions:
        return traj, target, torch.stack(act_hist, 1)  # actions [B,T]
    return traj, target


def cost(traj, target):
    c = traj[:, CONTROL_START_IDX:COST_END_IDX]
    tg = target[:, CONTROL_START_IDX:COST_END_IDX]
    lat = ((c - tg) ** 2).mean(1) * 100
    jerk = ((c[:, 1:] - c[:, :-1]) / DEL_T).pow(2).mean(1) * 100
    return lat, jerk, lat * LAT_ACCEL_COST_MULTIPLIER + jerk
