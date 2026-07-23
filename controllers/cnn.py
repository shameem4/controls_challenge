import os
import numpy as np
import torch
from pathlib import Path
from . import BaseController
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nets import make_net, build_ff_window

CKPT = os.environ.get('CNN_CKPT', 'cnn_D.pt')   # cnn_D = big residual/gating net, 51.34 on 5000
_ROOT = Path(__file__).resolve().parent.parent


class Controller(BaseController):
    """Eval wrapper for the trained preview CNN (runs on CPU, one step at a time).
    Default is the 'big' net (cnn_D). For base-arch checkpoints (cnn_B) set CNN_ARCH=base."""

    def __init__(self, ckpt=None, i_clip=5.0):
        self.net = make_net(os.environ.get('CNN_ARCH', 'big'))   # read at init (fork-safe)
        self.net.load_state_dict(torch.load(_ROOT / (ckpt or CKPT), map_location='cpu'))
        self.net.eval()
        self.integ = 0.0
        self.prev = 0.0
        self.i_clip = i_clip

    @torch.no_grad()
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        e = target_lataccel - current_lataccel
        self.integ = float(np.clip(self.integ + e, -self.i_clip, self.i_clip))
        fut_lat = np.asarray(future_plan.lataccel, dtype=np.float32)
        fut_roll = np.asarray(future_plan.roll_lataccel, dtype=np.float32)
        ff_win = build_ff_window(
            np.float32(target_lataccel), np.float32(state.roll_lataccel),
            fut_lat, fut_roll, np)                     # [H+1]
        ff_win = torch.from_numpy(ff_win.astype(np.float32))[None]
        v = torch.tensor([state.v_ego], dtype=torch.float32)
        fb = torch.tensor([[e, self.integ, self.prev]], dtype=torch.float32)
        out = self.net(ff_win, v, fb).item()
        self.prev = e
        return out
