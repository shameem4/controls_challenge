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
# and multi-horizon preview-error features [M]. Weights trained end-to-end via Gumbel rollouts
# with soft-token full BPTT (see torch_sim.py).
#
# Four interchangeable weight sets, identical architecture (cfg 'PM'):
#   cnn_v3.pt    45.74 on the full 5000  <- DEFAULT. Same recipe as cnn_v2, just trained to
#                convergence: 2700 iterations at ACC=4 (10,800 rollouts) against the released
#                checkpoints' 400 iterations (1,600). The prior checkpoints were simply
#                UNDERTRAINED. Beats cnn_v2 by -1.167 [-1.91,-0.64] on the headline 5000, median
#                -0.364, better on 3016/5000 (sign z=+14.6), and improves mean, median, p99 and
#                win-rate together -- the first cnn result here that does not trade median for
#                tail. See FINDINGS_GRAD_ARM.md.
#   cnn_v2.pt    46.91 on the full 5000. Plain policy optimisation, 1000 iters. Was the default
#                until cnn_v3; statistically tied with cnn_dual.pt.
#   cnn_dual.pt  46.89. Behaviour cloning onto ff_pi, then policy optimisation (dual_train.py).
#                Kept ONLY as evidence for a negative result: against cnn_v2 as a matched control
#                it is a coin flip (-0.016, CI [-0.61,+0.39], 2490/5000), so the BC->PO schedule
#                buys nothing. See FINDINGS_GAIN_PRIOR.md.
#   cnn_PM.pt    47.87. The v1-learned-47.87 tag; a below-average training run, kept so the earlier
#                published number stays reproducible.
# Run-to-run training variance on this architecture is ~1.0 point on the headline metric, which is
# larger than most effects worth chasing -- compare against a fresh matched control, never against
# whatever was shipped last.
_ROOT = Path(__file__).resolve().parent.parent
CKPT = os.environ.get('CNN_CKPT', 'cnn_v3.pt')
CFG = os.environ.get('CNN_CFG', 'PM')


class Controller(BaseController):
    """Eval wrapper for the AblNet preview controller (runs on CPU, one step at a time)."""

    def __init__(self, ckpt=None, cfg=None, i_clip=5.0, ch=None):
        self.cfg = CFG if cfg is None else cfg
        sd = torch.load(_ROOT / (ckpt or CKPT), map_location='cpu')
        # Infer the width from the checkpoint rather than assuming the default. Hardcoding ch=32
        # made every wider checkpoint fail to load with a size-mismatch wall of text.
        if ch is None:
            ch = sd['ff_conv.0.weight'].shape[0]
        self.net = AblNet(self.cfg, ch=ch, fb_hidden=ch, res_hidden=ch)
        self.net.load_state_dict(sd)
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
