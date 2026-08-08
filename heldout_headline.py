"""Honest headline for cnn_v4 on data it was never trained OR tuned on.

`cap_ab.py` trains on ALL[2000:4000] and validates on ALL[4000:4200], but the cnn headline has been
quoted on ALL[:5000] -- which CONTAINS the training range, a 40% overlap. `FINDINGS_BUGBASH.md`
found no evidence of memorisation via a matched control, but the test was underpowered, and arguing
about contamination is strictly worse than not having any. `controllers/ff_pi_rl2.py` already
reports both bases side by side for the classical arm; this does the same for the learned one.

ALL[5000:] is pristine: never trained on, never used for checkpoint selection, never tuned against.

Threads are pinned to 1 per worker. The default lets each torch worker grab ~3 cores, so 14 workers
oversubscribed the box by 3x and the first attempt at this crawled.
"""
import os
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'
import sys, numpy as np, importlib
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OFF = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
ARMS = [('cnn_v4', 'cnn', dict(ckpt='cnn_v4.pt')),
        ('ff_pi_tau', 'ff_pi_tau', {}),
        ('pid', 'pid', {})]
_M = None
def model():
    global _M
    if _M is None: _M = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    return _M
def one(f, mod, kw):
    import torch; torch.set_num_threads(1)
    C = importlib.import_module('controllers.' + mod).Controller
    r = TinyPhysicsSimulator(model(), str(f), controller=C(**kw), debug=False).rollout()
    return (r['total_cost'], r['lataccel_cost'], r['jerk_cost'])

if __name__ == '__main__':
    files = sorted(Path('data/SYNTHETIC').iterdir())[OFF:OFF + N]
    print(f'PRISTINE ALL[{OFF}:{OFF+N}]  n={len(files)}  (never trained, never tuned)', flush=True)
    print(f"  {'controller':12} {'mean':>9} {'median':>9} {'lat':>8} {'jerk':>8} {'sem':>7}", flush=True)
    out = {}
    for name, mod, kw in ARMS:
        a = np.array(process_map(partial(one, mod=mod, kw=kw), files, max_workers=14,
                                 chunksize=8, disable=True))
        out[name] = a
        print(f'  {name:12} {a[:,0].mean():9.3f} {np.median(a[:,0]):9.3f} '
              f'{50*a[:,1].mean():8.3f} {a[:,2].mean():8.3f} '
              f'{a[:,0].std(ddof=1)/np.sqrt(len(a)):7.3f}', flush=True)
    rng = np.random.default_rng(0)
    d = out['cnn_v4'][:, 0] - out['ff_pi_tau'][:, 0]
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10000)])
    print(f"\n  cnn_v4 - ff_pi_tau, paired: {d.mean():+.3f}  "
          f"95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]  "
          f"cnn better on {int((d<0).sum())}/{len(d)}", flush=True)
