"""Closed-loop setpoint-step responses for the REAL controllers, across operating conditions.

`step_plot.py` swept PID gains on a bare rig. This runs the actual shipped controllers -- which use
25 steps of preview -- so the synthetic scenarios must supply a real `future_plan`. They are
therefore written as CSVs and driven through the real `TinyPhysicsSimulator`, meaning every
controller behaves exactly as it does on the benchmark, preview and all.

Noise realisations come for free: the simulator seeds from md5(filename), so N copies of the same
profile under different names give N independent draws. Curves are the mean with a +-1 s.e. band.

What to look for. `pid` and `pid_boot` are causal-only, so they cannot move before the step. `ff_pi_boot`
and `cnn` see the step coming through the preview window and should begin turning EARLY -- that lead
is most of what separates them, and it is invisible in any open-loop plant test.

Conditions sweep speed, road roll, longitudinal accel, and step size (gentle lane adjustment through
to hard-corner entry). The operating point is placed at equilibrium (`u0 = (c0 - roll)/G(v)`), since
starting off-equilibrium leaves the plant mid-transient and the response then rides on a drift -- an
error that produced 253% "overshoot" in an earlier bench test.

Usage: python step_grid.py [out.png]     env: NREP
"""
import os, sys, shutil, numpy as np, pandas as pd
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, DEL_T, ACC_G

NREP = int(os.environ.get('NREP', 48))
PRE, HOLD, POST = 25, 45, 45
T = CONTROL_START_IDX + PRE + HOLD + POST
TMP = Path('/tmp/step_grid')
GAIN_FIT = np.load(Path(__file__).resolve().parent / 'gain_fit.npy')

CONDS = [
    ('v=8 m/s (low speed)',      dict(v=8.0,  roll=0.0,  a=0.0, step=1.0)),
    ('v=22 m/s (nominal)',       dict(v=22.0, roll=0.0,  a=0.0, step=1.0)),
    ('v=34 m/s (high speed)',    dict(v=34.0, roll=0.0,  a=0.0, step=1.0)),
    ('roll = -1.0 (off-camber)', dict(v=22.0, roll=-1.0, a=0.0, step=1.0)),
    ('roll = +1.0',              dict(v=22.0, roll=1.0,  a=0.0, step=1.0)),
    ('a_ego = +2 (accelerating)', dict(v=22.0, roll=0.0, a=2.0, step=1.0)),
    ('gentle step (0.5)',        dict(v=22.0, roll=0.0,  a=0.0, step=0.5)),
    ('hard corner entry (2.5)',  dict(v=22.0, roll=0.0,  a=0.0, step=2.5)),
]

CTRLS = [
    ('pid (stock)',   'pid',        {}),
    ('pid_boot',      'pid_boot',   {}),
    ('ff_pi_boot',    'ff_pi_boot', dict(boot=0.005)),
    ('cnn_v2',        'cnn',        dict(ckpt='cnn_v2.pt')),
]


def write_case(name, v, roll, a, step, n):
    """n identical synthetic segments under different filenames -> n independent noise draws."""
    d = TMP / name
    d.mkdir(parents=True, exist_ok=True)
    tgt = np.zeros(T); t0 = CONTROL_START_IDX + PRE
    tgt[t0:t0 + HOLD] = step
    G = float(np.clip(np.polyval(GAIN_FIT, v), 0.3, 4.0))
    u0 = (0.0 - roll) / G                                     # equilibrium steer for c=0 pre-step
    df = pd.DataFrame(dict(roll=np.full(T, np.arcsin(np.clip(roll / ACC_G, -1, 1))),
                           vEgo=np.full(T, v), aEgo=np.full(T, a),
                           targetLateralAcceleration=tgt,
                           steerCommand=np.full(T, -u0)))
    files = []
    for i in range(n):
        f = d / f'{i:05d}.csv'
        df.to_csv(f, index=False)
        files.append(str(f))
    return files, tgt, t0


_M = [None]
def _model():
    if _M[0] is None:
        _M[0] = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    return _M[0]


def _run(f, mod, kw):
    import importlib
    C = importlib.import_module('controllers.' + mod).Controller
    sim = TinyPhysicsSimulator(_model(), f, controller=C(**kw), debug=False)
    sim.rollout()
    return np.array(sim.current_lataccel_history)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'step_grid.png'
    if TMP.exists():
        shutil.rmtree(TMP)
    fig, axes = plt.subplots(2, 4, figsize=(19, 8), sharex=True)
    for ax, (title, c) in zip(axes.ravel(), CONDS):
        files, tgt, t0 = write_case(title.split()[0].replace('=', '') + str(abs(hash(title)) % 999),
                                    c['v'], c['roll'], c['a'], c['step'], NREP)
        tt = (np.arange(T) - t0) * DEL_T
        ax.plot(tt, tgt, 'k--', lw=1.3, label='target', zorder=5)
        for lbl, mod, kw in CTRLS:
            Y = np.array(process_map(partial(_run, mod=mod, kw=kw), files,
                                     max_workers=24, chunksize=2, disable=True))
            n = min(Y.shape[1], T)
            mu, se = Y[:, :n].mean(0), Y[:, :n].std(0) / np.sqrt(len(Y))
            ax.plot(tt[:n], mu, lw=1.6, label=lbl)
            ax.fill_between(tt[:n], mu - se, mu + se, alpha=0.16, lw=0)
        ax.axvline(0, color='0.6', lw=0.8, ls=':')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(-2.0, (HOLD + POST) * DEL_T - 2.0)
        print(f'  done: {title}', flush=True)
    for ax in axes[1]:
        ax.set_xlabel('time from step (s)')
    for ax in axes[:, 0]:
        ax.set_ylabel('lateral acceleration (m/s²)')
    axes[0, 0].legend(fontsize=8, loc='lower right')
    fig.suptitle(f'Closed-loop setpoint step, real controllers, mean of {NREP} noise realisations (±1 s.e.). '
                 f'ff_pi_boot and cnn see the step coming via 25-step preview; pid variants cannot.',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
