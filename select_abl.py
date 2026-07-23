"""Real-sim checkpoint selection for an ablation config on held-out [1200:1320].
Usage: python select_abl.py <base|P|M|H|PMH> [nseg]"""
import sys, os, glob, shutil, numpy as np
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.cnn import Controller

cfg_arg = sys.argv[1]; cfg = '' if cfg_arg == 'base' else cfg_arg
nseg = int(sys.argv[2]) if len(sys.argv) > 2 else 120
files = sorted(Path('data/SYNTHETIC').iterdir())[1200:1200 + nseg]
_M = [None]
def m():
    if _M[0] is None: _M[0] = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    return _M[0]
def run(f, ckpt):
    c = Controller(ckpt=ckpt, cfg=cfg)
    return TinyPhysicsSimulator(m(), str(f), controller=c, debug=False).rollout()['total_cost']
def ev(ckpt):
    r = process_map(partial(run, ckpt=ckpt), files, max_workers=12, chunksize=10, disable=True)
    return float(np.mean(r))

best = (1e9, None)
for ck in sorted(glob.glob(f'ckpts/abl{cfg_arg}_[0-9]*.pt')):
    c = ev(ck)
    if c < best[0]: best = (c, ck)
shutil.copy(best[1], f'ckpts/abl{cfg_arg}_BEST.pt')
print(f"[{cfg_arg}] BEST real {os.path.basename(best[1])} = {best[0]:.2f} -> ckpts/abl{cfg_arg}_BEST.pt", flush=True)
