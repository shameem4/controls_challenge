"""Classic closed-loop setpoint-step plots for this plant: target steps, controller tracks.

This is the textbook tuning picture -- step the SETPOINT, watch the response go sluggish / good /
oscillatory as a gain is swept. It is the closed-loop counterpart to `sysid.py`, which steps the
ACTION to identify the plant itself (gain, dead time, time constant). Both are needed: the open-loop
test tells you what the plant is, the closed-loop test tells you what your tuning does.

Conditions are synthesised (constant v_ego, roll, a_ego) so the only thing varying is the setpoint
and the gains. The plant samples, so each curve is the mean over NREP independent noise realisations
with a +-1 s.e. band -- without that the step response is buried in the random walk.

Measured plant, for reference: K ~ 1.5, dead time L ~ 0.25 s, T ~ 0.15 s, so L/T ~ 1.7 and the loop
is DEAD-TIME DOMINATED. Classical theory predicts that aggressive gains will ring, and that is what
the sweep shows.

Usage: python step_plot.py [out.png]     env: NREP, V, STEP
"""
import os, sys, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_sim import Plant
from sysid import Rig, synth
from tinyphysics import CONTROL_START_IDX, DEL_T

DEV = 'cuda'
NREP = int(os.environ.get('NREP', 96))
V = float(os.environ.get('V', 22.0))
STEP = float(os.environ.get('STEP', 1.0))
PRE, HOLD, POST = 30, 60, 60


def target_profile(T):
    """Flat, step up, step back down -- the shape in every textbook tuning figure."""
    tgt = np.zeros(T)
    t0 = CONTROL_START_IDX + PRE
    tgt[t0:t0 + HOLD] = STEP
    return tgt, t0


@torch.no_grad()
def run_pid(plant, p, i, d, T, tgt, seed=0):
    """Closed-loop response of comma's discrete PID form: u = p*e + i*sum(e) + d*(e - e_prev)."""
    torch.manual_seed(seed)
    segs = synth(V, 0.0, 0.0, 0.0, 0.0, NREP, T)
    R = Rig(plant, segs, T)
    tg = torch.tensor(tgt, dtype=torch.float32, device=DEV)
    integ = torch.zeros(NREP, device=DEV)
    prev = torch.zeros(NREP, device=DEV)
    out = []
    while R.t < T - 1:
        t = R.t
        e = tg[t] - R.cur
        integ = integ + e
        u = p * e + i * integ + d * (e - prev)
        prev = e
        out.append(R.step(u))
    return torch.stack(out, 1).cpu().numpy()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'step_response.png'
    plant = Plant(device=DEV)
    T = CONTROL_START_IDX + PRE + HOLD + POST
    tgt, t0 = target_profile(T)
    tt = (np.arange(T - 1 - 20) - (t0 - 20)) * DEL_T          # seconds relative to the step

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    plots = [
        ('proportional gain  (i=0.10, d=0)', [('p=0.05', 0.05, 0.10, 0.0), ('p=0.195 (stock)', 0.195, 0.10, 0.0),
                                              ('p=0.48 (Ziegler-Nichols)', 0.48, 0.10, 0.0), ('p=0.90', 0.90, 0.10, 0.0)]),
        ('integral gain  (p=0.195, d=0)', [('i=0.00', 0.195, 0.0, 0.0), ('i=0.05', 0.195, 0.05, 0.0),
                                           ('i=0.10 (stock)', 0.195, 0.10, 0.0), ('i=0.25', 0.195, 0.25, 0.0)]),
        ('derivative gain  (p=0.195, i=0.10)', [('d=-0.053 (stock)', 0.195, 0.10, -0.053), ('d=0', 0.195, 0.10, 0.0),
                                                ('d=+0.20', 0.195, 0.10, 0.20), ('d=+0.60 (ZN)', 0.195, 0.10, 0.60)]),
    ]
    for ax, (title, cfgs) in zip(axes, plots):
        ax.plot(tt, tgt[20:T - 1], 'k--', lw=1.2, label='target', zorder=5)
        for lbl, p, i, d in cfgs:
            y = run_pid(plant, p, i, d, T, tgt)
            mu, se = y.mean(0), y.std(0) / np.sqrt(NREP)
            ax.plot(tt, mu, lw=1.6, label=lbl)
            ax.fill_between(tt, mu - se, mu + se, alpha=0.18, lw=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('time from step (s)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlim(-1.5, (HOLD + POST) * DEL_T - 1.5)
    axes[0].set_ylabel('lateral acceleration (m/s²)')
    fig.suptitle(f'Closed-loop setpoint step, v={V:.0f} m/s, mean of {NREP} noise realisations (±1 s.e.)  '
                 f'— plant: K≈1.5, L≈0.25 s, T≈0.15 s, L/T≈1.7 (dead-time dominated)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
