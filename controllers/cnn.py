import os
import numpy as np
import torch
from pathlib import Path
from . import BaseController
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nets import AblNet, build_ff_window, build_multihorizon, V_SCALE

# Deliverable: the "PM" preview net — feedforward conv over the (target-roll) preview,
# gain-scheduled by v_ego, feedback head + criticality-gated residual, and (the two
# ideas distilled from jonoomph's ML_PID that helped us) previous-action inputs [P]
# and multi-horizon preview-error features [M]. Weights trained two-stage (deterministic
# base -> noisy fine-tune). 49.70 on the full 5000 (PID 110.76).
_ROOT = Path(__file__).resolve().parent.parent
CKPT = os.environ.get('CNN_CKPT', 'cnn_PM.pt')
CFG = os.environ.get('CNN_CFG', 'PM')


class Controller(BaseController):
    """Eval wrapper for the AblNet preview controller (runs on CPU, one step at a time)."""

    def __init__(self, ckpt=None, cfg=None, i_clip=5.0):
        self.cfg = CFG if cfg is None else cfg
        self.net = AblNet(self.cfg)
        self.net.load_state_dict(torch.load(_ROOT / (ckpt or CKPT), map_location='cpu'))
        self.net.eval()
        self.integ = 0.0
        self.prev = 0.0
        self.pact = [0.0] * self.net.n_prev
        self.hist = []
        self.i_clip = i_clip

    @torch.no_grad()
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        e = target_lataccel - current_lataccel
        self.integ = float(np.clip(self.integ + e, -self.i_clip, self.i_clip))
        fl = np.asarray(future_plan.lataccel, dtype=np.float32)
        fr = np.asarray(future_plan.roll_lataccel, dtype=np.float32)
        ff_win = build_ff_window(np.float32(target_lataccel), np.float32(state.roll_lataccel), fl, fr, np)
        feats = [e, self.integ, self.prev]
        if 'P' in self.cfg:
            feats = feats + self.pact[-self.net.n_prev:]
        fb = np.array(feats, dtype=np.float32)
        if 'M' in self.cfg:
            mh = build_multihorizon(np.float32(current_lataccel), np.float32(state.roll_lataccel), fl, fr, np)
            fb = np.concatenate([fb, mh])
        hist_t = None
        if 'H' in self.cfg:
            self.hist.append(np.array([e, state.roll_lataccel, state.v_ego / V_SCALE, state.a_ego], dtype=np.float32))
            buf = self.hist[-self.net.hist_w:]
            if len(buf) < self.net.hist_w:
                buf = [np.zeros(4, dtype=np.float32)] * (self.net.hist_w - len(buf)) + buf
            hist_t = torch.from_numpy(np.stack(buf))[None].transpose(1, 2)
        out = float(self.net(torch.from_numpy(ff_win)[None],
                             torch.tensor([state.v_ego], dtype=torch.float32),
                             torch.from_numpy(fb)[None], hist_t).item())
        self.pact.append(out)
        self.prev = e
        return out
