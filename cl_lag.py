"""CLOSED-LOOP tracking lag: how far behind the target does the achieved lataccel actually sit?

The open-loop step response (5-7 steps to settle) assumed a single held action. In closed loop the
controller re-acts every step, so the effective response is faster. This measures the lag directly:
for each controller, find the shift k that maximises corr(achieved[t], target[t-k]).

  k > 0  achieved LAGS the target by k steps
  k ~ 0  the loop is tracking in phase
  k < 0  the loop is LEADING the target
"""
import numpy as np, importlib
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map
from tinyphysics import (TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, COST_END_IDX)
SEG=sorted(Path('data/SYNTHETIC').iterdir())[500:700]
_M=[None]
def m():
    if _M[0] is None: _M[0]=TinyPhysicsModel('models/tinyphysics.onnx',debug=False)
    return _M[0]
def run(f,kind,kw):
    C=importlib.import_module(f'controllers.{kind}').Controller
    s=TinyPhysicsSimulator(m(),str(f),controller=C(**kw),debug=False); s.rollout()
    import pandas as pd
    lat=np.array(s.current_lataccel_history[CONTROL_START_IDX:COST_END_IDX])
    tgt=pd.read_csv(f)['targetLateralAcceleration'].values[CONTROL_START_IDX:COST_END_IDX]
    best=(None,-9)
    for k in range(-4,9):                                  # negative = leading
        a=lat[max(k,0):len(lat)+min(k,0)]
        b=tgt[max(-k,0):len(tgt)+min(-k,0)]
        if len(a)<50: continue
        c=np.corrcoef(a,b)[0,1]
        if c>best[1]: best=(k,c)
    return best[0], best[1], np.sqrt(np.mean((lat-tgt)**2))
ARMS=[('stock PID           ','pid',{}),
      ('PID + lookahead k=2 ','pid_look',{'k0':2.0}),
      ('PID + v-sched t90x.4','pid_phys',{'basis':'t90','scale':0.4}),
      ('ff_pi_rl2           ','ff_pi_rl2',{}),
      ('cnn                 ','cnn',{})]
print(f"{'controller':22} {'best lag':>9} {'corr':>7} {'RMS err':>9}")
for lbl,kind,kw in ARMS:
    r=np.array(process_map(partial(run,kind=kind,kw=kw),SEG,max_workers=24,chunksize=4,disable=True))
    print(f'{lbl} {r[:,0].mean():8.2f}  {r[:,1].mean():7.4f} {r[:,2].mean():9.4f}',flush=True)
