"""Real-sim checkpoint selection: eval every ckpts/<tag>_*.pt on the REAL numpy sim
(held-out VAL segments) and report the best. Usage: python select_real.py <tag> <arch> [nseg]
"""
import sys, os, glob, numpy as np
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.cnn import Controller

TAG = sys.argv[1]
ARCH = sys.argv[2] if len(sys.argv) > 2 else 'base'
NSEG = int(sys.argv[3]) if len(sys.argv) > 3 else 120
os.environ['CNN_ARCH'] = ARCH
# VAL range disjoint from TRAIN[2000:2600]; use a held-out slice not used for torch val selection
files = sorted(Path('data/SYNTHETIC').iterdir())[1200:1200 + NSEG]
_M = None
def model():
    global _M
    if _M is None: _M = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    return _M

def run_one(f, ckpt):
    c = Controller(ckpt=ckpt)
    return TinyPhysicsSimulator(model(), str(f), controller=c, debug=False).rollout()['total_cost']

def ev(ckpt):
    r = process_map(partial(run_one, ckpt=ckpt), files, max_workers=12,
                    chunksize=max(1, len(files)//12), disable=True)
    return float(np.mean(r))

ckpts = sorted(glob.glob(f'ckpts/{TAG}_[0-9]*.pt'))
print(f"real-sim selection tag={TAG} arch={ARCH} on {NSEG} held-out segs [1200:{1200+NSEG}]")
best = (1e9, None)
for ck in ckpts:
    c = ev(ck)
    star = ''
    if c < best[0]: best = (c, ck); star = ' *'
    print(f"  {os.path.basename(ck):<24} real_val={c:7.2f}{star}", flush=True)
print(f"BEST real: {best[1]}  = {best[0]:.2f}")
