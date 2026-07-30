"""Stage 3: tradition-structured preview CNN controller.

steer = feedforward(conv over (ref-roll) preview window, gain-scheduled by v)
      + feedback(PI-like head on tracking error / integral)

Feature construction is defined once here and reused by both the torch training
rollout and the numpy eval controller (controllers/cnn.py) to avoid train/serve skew.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

GAIN_FIT = np.load(Path(__file__).resolve().parent / 'gain_fit.npy')  # measured steer->lataccel DC gain


def plant_gain(v):
    """Measured G(v), batched torch. Horner over the fitted quadratic; same clip as ff_pi."""
    g = torch.zeros_like(v)
    for c in GAIN_FIT:
        g = g * v + float(c)
    return g.clamp(0.3, 4.0)

H = 25            # preview horizon (steps ahead)
V_SCALE = 30.0    # v_ego normalization


class PreviewCNN(nn.Module):
    def __init__(self, h=H, ch=16, fb_hidden=16):
        super().__init__()
        self.h = h
        self.ff_conv = nn.Sequential(
            nn.Conv1d(1, ch, 5, padding=2), nn.Tanh(),
            nn.Conv1d(ch, ch, 5, padding=2), nn.Tanh())
        self.ff_head = nn.Linear(ch * (h + 1), 1)
        self.film = nn.Linear(1, 2)                       # gain schedule from v
        self.fb = nn.Sequential(nn.Linear(3, fb_hidden), nn.Tanh(),
                                nn.Linear(fb_hidden, 1))
        # init feedback + film near-zero so untrained net ~ passthrough feedforward
        nn.init.zeros_(self.fb[-1].weight); nn.init.zeros_(self.fb[-1].bias)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, ff_win, v, fb_feats):
        # ff_win [B,H+1], v [B], fb_feats [B,3]
        x = self.ff_conv(ff_win.unsqueeze(1)).flatten(1)
        ff = self.ff_head(x).squeeze(-1)
        s, sh = self.film((v / V_SCALE).unsqueeze(-1)).unbind(-1)
        ff = ff * (1 + s) + sh
        fb = self.fb(fb_feats).squeeze(-1)
        return ff + fb


class PreviewCNNBig(nn.Module):
    """Bigger net: base 2-DOF (as PreviewCNN) + a residual head gated by a criticality
    signal (error magnitude, preview span, peak slope) computed internally from the same
    features -- so forward() signature is identical and eval/rollout code is unchanged."""
    def __init__(self, h=H, ch=32, fb_hidden=32, res_hidden=32):
        super().__init__()
        self.h = h
        self.ff_conv = nn.Sequential(
            nn.Conv1d(1, ch, 5, padding=2), nn.Tanh(),
            nn.Conv1d(ch, ch, 5, padding=2), nn.Tanh(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.Tanh())
        self.ff_head = nn.Linear(ch * (h + 1), 1)
        self.film = nn.Linear(1, 2)
        self.fb = nn.Sequential(nn.Linear(3, fb_hidden), nn.Tanh(), nn.Linear(fb_hidden, 1))
        self.res = nn.Sequential(nn.Linear(6, res_hidden), nn.Tanh(),
                                 nn.Linear(res_hidden, res_hidden), nn.Tanh(),
                                 nn.Linear(res_hidden, 1))
        self.gate = nn.Linear(3, 1)                      # criticality -> gate
        for m in (self.fb[-1], self.res[-1]):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, ff_win, v, fb_feats):
        x = self.ff_conv(ff_win.unsqueeze(1)).flatten(1)
        ff = self.ff_head(x).squeeze(-1)
        s, sh = self.film((v / V_SCALE).unsqueeze(-1)).unbind(-1)
        ff = ff * (1 + s) + sh
        fb = self.fb(fb_feats).squeeze(-1)
        # criticality stats from the preview window + tracking error
        span = ff_win.max(1).values - ff_win.min(1).values
        slope = (ff_win[:, 1:] - ff_win[:, :-1]).abs().max(1).values
        emag = fb_feats[:, 0].abs()
        crit = torch.stack([span, slope, emag], -1)
        gate = torch.sigmoid(self.gate(crit)).squeeze(-1)
        res = self.res(torch.cat([fb_feats, crit], -1)).squeeze(-1)
        return ff + fb + gate * res


def make_net(arch='base'):
    return PreviewCNNBig() if arch == 'big' else PreviewCNN()


def build_ff_window(target_t, roll_t, fut_lat, fut_roll, xp):
    """(ref - roll) preview window of length H+1, edge-padded when future is short."""
    def tail(head, fut):
        n = fut.shape[-1]
        if n >= H:
            return fut[..., :H]
        if xp is torch:
            if n == 0:
                return head.unsqueeze(-1).expand(-1, H)
            return torch.cat([fut, fut[..., -1:].expand(-1, H - n)], -1)
        if n == 0:
            return np.repeat(head[..., None], H, -1)
        return np.concatenate([fut, np.repeat(fut[..., -1:], H - n, -1)], -1)
    tl = tail(target_t, fut_lat); rl = tail(roll_t, fut_roll)
    if xp is torch:
        return torch.cat([(target_t - roll_t).unsqueeze(-1), tl - rl], -1)
    return np.concatenate([(target_t - roll_t)[..., None], tl - rl], -1)


class TorchPolicy:
    """Stateful CNN controller for the batched torch rollout. Maintains PI state."""
    def __init__(self, net, B, dev, i_clip=5.0):
        self.net = net; self.integ = torch.zeros(B, device=dev)
        self.prev = torch.zeros(B, device=dev); self.i_clip = i_clip

    def detach_state(self):
        self.integ = self.integ.detach(); self.prev = self.prev.detach()

    def __call__(self, ctx):
        e = ctx['target'] - ctx['cur']
        self.integ = (self.integ + e).clamp(-self.i_clip, self.i_clip)
        ff_win = build_ff_window(ctx['target'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
        fb = torch.stack([e, self.integ, self.prev], -1)
        out = self.net(ff_win, ctx['v'], fb)
        self.prev = e
        return out


# ----- Ablation: 3 borrowable ideas from jonoomph's ML_PID -----
# P = feed previous actions ; M = multi-horizon preview-error features ; H = past-history temporal branch
MH_HORIZONS = [(0, 3), (3, 8), (8, 15)]   # near / mid / far
HIST_W = 20
N_PREV = 3


def build_multihorizon(cur, roll, fut_lat, fut_roll, xp):
    """6 features: (cur - mean(future_lataccel)) and (roll - mean(future_roll)) at 3 horizons."""
    def hmean(fut, lo, hi, fallback):
        seg = fut[..., lo:hi]
        if seg.shape[-1] == 0:
            return fallback
        return seg.mean(-1)
    feats = ([cur - hmean(fut_lat, lo, hi, cur) for lo, hi in MH_HORIZONS] +
             [roll - hmean(fut_roll, lo, hi, roll) for lo, hi in MH_HORIZONS])
    return torch.stack(feats, -1) if xp is torch else np.array(feats, dtype=np.float32)


class AblNet(nn.Module):
    def __init__(self, cfg='', h=H, ch=32, fb_hidden=32, res_hidden=32, hist_ch=16):
        super().__init__()
        self.cfg = cfg; self.n_prev = N_PREV; self.hist_w = HIST_W
        self.ff_conv = nn.Sequential(
            nn.Conv1d(1, ch, 5, padding=2), nn.Tanh(),
            nn.Conv1d(ch, ch, 5, padding=2), nn.Tanh(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.Tanh())
        self.ff_head = nn.Linear(ch * (h + 1), 1)
        self.film = nn.Linear(1, 2)
        fb_dim = 3 + (N_PREV if 'P' in cfg else 0) + (6 if 'M' in cfg else 0)
        self.hist_ch = hist_ch
        if 'H' in cfg:
            self.hist_conv = nn.Sequential(nn.Conv1d(4, hist_ch, 5, padding=2), nn.Tanh())
            fb_dim += hist_ch
        self.fb = nn.Sequential(nn.Linear(fb_dim, fb_hidden), nn.Tanh(), nn.Linear(fb_hidden, 1))
        self.res = nn.Sequential(nn.Linear(fb_dim, res_hidden), nn.Tanh(),
                                 nn.Linear(res_hidden, res_hidden), nn.Tanh(), nn.Linear(res_hidden, 1))
        self.gate = nn.Linear(3, 1)
        for m in (self.fb[-1], self.res[-1]):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, ff_win, v, fb_feats, hist=None):
        if 'G' in self.cfg:
            # preview window in STEERING units: the measured gain schedule replaces what `film`
            # would otherwise have to discover. film is retained below as a learned correction.
            ff_win = ff_win / plant_gain(v).unsqueeze(-1)
        x = self.ff_conv(ff_win.unsqueeze(1)).flatten(1)
        ff = self.ff_head(x).squeeze(-1)
        s, sh = self.film((v / V_SCALE).unsqueeze(-1)).unbind(-1)
        ff = ff * (1 + s) + sh
        fbin = fb_feats
        if 'H' in self.cfg:
            fbin = torch.cat([fbin, self.hist_conv(hist).mean(-1)], -1)
        fb = self.fb(fbin).squeeze(-1)
        span = ff_win.max(1).values - ff_win.min(1).values
        slope = (ff_win[:, 1:] - ff_win[:, :-1]).abs().max(1).values
        crit = torch.stack([span, slope, fb_feats[:, 0].abs()], -1)
        gate = torch.sigmoid(self.gate(crit)).squeeze(-1)
        return ff + fb + gate * self.res(fbin).squeeze(-1)


class AblPolicy:
    """Stateful torch-rollout controller for AblNet; maintains PI + prev-actions + history state."""
    def __init__(self, net, B, dev, i_clip=5.0):
        self.net = net; self.cfg = net.cfg; self.B = B; self.dev = dev; self.i_clip = i_clip
        self.integ = torch.zeros(B, device=dev); self.prev = torch.zeros(B, device=dev)
        self.pact = [torch.zeros(B, device=dev) for _ in range(net.n_prev)]
        self.hist = []

    def detach_state(self):
        self.integ = self.integ.detach(); self.prev = self.prev.detach()
        self.pact = [a.detach() for a in self.pact]
        self.hist = [h.detach() for h in self.hist]

    def __call__(self, ctx):
        e = ctx['target'] - ctx['cur']
        self.integ = (self.integ + e).clamp(-self.i_clip, self.i_clip)
        ff_win = build_ff_window(ctx['target'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)
        feats = [e, self.integ, self.prev]
        if 'P' in self.cfg:
            feats = feats + self.pact[-self.net.n_prev:]
        fb = torch.stack(feats, -1)
        if 'M' in self.cfg:
            fb = torch.cat([fb, build_multihorizon(ctx['cur'], ctx['roll'], ctx['fut_lat'], ctx['fut_roll'], torch)], -1)
        hist = None
        if 'H' in self.cfg:
            self.hist.append(torch.stack([e, ctx['roll'], ctx['v'] / V_SCALE, ctx['a']], -1))
            buf = self.hist[-self.net.hist_w:]
            if len(buf) < self.net.hist_w:
                buf = [torch.zeros(self.B, 4, device=self.dev)] * (self.net.hist_w - len(buf)) + buf
            hist = torch.stack(buf, 1).transpose(1, 2)                 # [B,4,W]
        out = self.net(ff_win, ctx['v'], fb, hist)
        self.pact.append(out.detach())
        self.prev = e
        return out
