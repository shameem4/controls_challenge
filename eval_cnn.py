"""Honest held-out eval on the REAL numpy sim: cnn vs ff_pi vs pid."""
import sys, os, numpy as np, importlib
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

OFFSET = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
files = sorted(Path('data/SYNTHETIC').iterdir())[OFFSET:OFFSET + N]
_M = None
def model():
    global _M
    if _M is None: _M = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    return _M

def run_one(f, name):
    c = importlib.import_module(f'controllers.{name}').Controller()
    r = TinyPhysicsSimulator(model(), str(f), controller=c, debug=False).rollout()
    return (r['lataccel_cost'], r['jerk_cost'], r['total_cost'])

def ev(name):
    a = np.array(process_map(partial(run_one, name=name), files, max_workers=12,
                             chunksize=max(1, len(files)//12), disable=True))
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()

print(f"REAL numpy sim, held-out segs [{OFFSET}:{OFFSET+N}]  (CNN_CKPT={os.environ.get('CNN_CKPT','cnn_A.pt')})")
for name in ['pid', 'ff_pi', 'cnn']:
    lat, jerk, tot = ev(name)
    print(f"  {name:<7} lat={lat:6.3f} jerk={jerk:6.2f} total={tot:7.2f}", flush=True)
