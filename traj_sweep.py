"""Sweep pid_traj against the real stochastic sim. Reports lat/jerk/total separately, because the
expected trade for outer-loop position control is worse tracking bought with smoother steering."""
import sys, itertools, numpy as np
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
import importlib, os
Controller = importlib.import_module("controllers." + os.environ.get("CTRL", "pid_traj")).Controller

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OFF = int(sys.argv[3]) if len(sys.argv) > 3 else 0
files = sorted(Path('data/SYNTHETIC').iterdir())[OFF:OFF + N]
_M = None
def model():
    global _M
    if _M is None: _M = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    return _M
def run_one(f, params):
    r = TinyPhysicsSimulator(model(), str(f), controller=Controller(**params), debug=False).rollout()
    return (r['lataccel_cost'], r['jerk_cost'], r['total_cost'])
def evaluate(params):
    a = np.array(process_map(partial(run_one, params=params), files, max_workers=14,
                             chunksize=max(1, len(files)//14), disable=True))
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), np.median(a[:, 2])

grid = eval(sys.argv[2]) if len(sys.argv) > 2 else {'w': [0.0, 1.0]}
# A dict of lists is a cross product; a list of dicts is an explicit set of points, which is what
# you want when parameters must move together (e.g. holding DC gain fixed while tau varies).
points = grid if isinstance(grid, list) else [
    dict(zip(grid, c)) for c in itertools.product(*grid.values())]
print(f"segs={N} offset={OFF}")
print(f"{'params':<52} {'lat':>7} {'jerk':>7} {'TOTAL':>9} {'median':>9}")
best = None
for p in points:
    lat, jerk, tot, med = evaluate(p)
    print(f"{' '.join(f'{k}={v}' for k, v in p.items()):<52} {lat:7.3f} {jerk:7.2f} {tot:9.2f} {med:9.2f}",
          flush=True)
    if best is None or tot < best[0]: best = (tot, p)
print(f"\nBEST total={best[0]:.2f}  {best[1]}")
