"""Coarse hyperparameter sweep for ff_pi against the real (stochastic) sim."""
import sys, numpy as np
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.ff_pi import Controller

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
files = sorted(Path('data/SYNTHETIC').iterdir())[:N]
_M = None
def model():
    global _M
    if _M is None:
        _M = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    return _M

def run_one(f, params):
    c = Controller(**params)
    sim = TinyPhysicsSimulator(model(), str(f), controller=c, debug=False)
    r = sim.rollout()
    return (r['lataccel_cost'], r['jerk_cost'], r['total_cost'])

def evaluate(params):
    res = process_map(partial(run_one, params=params), files, max_workers=16,
                      chunksize=max(1, len(files)//16), disable=True)
    a = np.array(res)
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()

# grid defined via argv[2] = comma-separated "key=val" overrides applied on top of a base grid spec
import itertools
grid = eval(sys.argv[2]) if len(sys.argv) > 2 else {'gain_scale': [0.4, 0.6, 0.8, 1.0]}
base = dict(kp=0.10, ki=0.05, lead=1, smooth=True)
keys = list(grid.keys())
print(f"segs={N}  base={base}")
best = None
for combo in itertools.product(*[grid[k] for k in keys]):
    p = dict(base); p.update(dict(zip(keys, combo)))
    lat, jerk, tot = evaluate(p)
    tag = " ".join(f"{k}={p[k]}" for k in keys)
    print(f"{tag:<40} lat={lat:6.3f} jerk={jerk:6.2f} total={tot:7.2f}", flush=True)
    if best is None or tot < best[0]:
        best = (tot, p)
print(f"\nBEST total={best[0]:.2f}  params={best[1]}")
