"""Find a CLEAN segment where pid genuinely holds the tightest line despite costing more.

06585 was used for this and turned out to be corrupted: 27 target steps exceed the plant's rate
limit, up to 9.1x, and ~80% of its cost comes from ~2 s of unfollowable data. 0 of 495 sampled
segments have more than 5 such steps, so it is not representative of anything.

Screens on the quantity the page actually DRAWS -- the 3 s leaky lane offset, matching
drive_template.html:laneOffsets -- not open-loop cross-track, which is a different signal.
"""
import numpy as np, pandas as pd, importlib
from pathlib import Path
from multiprocessing import Pool
from tinyphysics import (TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX,
                         COST_END_IDX, DEL_T, MAX_ACC_DELTA)
lo, hi = CONTROL_START_IDX, COST_END_IDX
ARMS = [('pid','pid',{}), ('ff_pi_boot','ff_pi_boot',dict(boot=0.005)),
        ('cnn_v4','cnn',dict(ckpt='cnn_v4.pt'))]
M = None
def init():
    global M; M = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    import torch; torch.set_num_threads(1)
def leaky(lat, tgt, v, tau=3.0):
    dec = np.exp(-DEL_T/tau); psi = y = 0.0; out = np.empty(len(tgt))
    for i in range(len(tgt)):
        vv = max(v[i], 1e-3)
        psi = psi*dec + ((lat[i]-tgt[i])/vv)*DEL_T
        y = y*dec + vv*psi*DEL_T
        out[i] = y
    return out
def job(f):
    df = pd.read_csv(f)
    tgt = df['targetLateralAcceleration'].values[lo:hi]; v = df['vEgo'].values[lo:hi]
    if v.min() < 5: return None
    if (np.abs(np.diff(tgt)) > MAX_ACC_DELTA).any(): return None      # clean segments only
    out = {'id': f.stem, 'peak': float(np.abs(tgt).max()), 'v': float(v.mean())}
    for key, mod, kw in ARMS:
        C = importlib.import_module('controllers.'+mod).Controller
        sim = TinyPhysicsSimulator(M, str(f), controller=C(**kw), debug=False)
        c = sim.rollout()
        lat = np.array(sim.current_lataccel_history[lo:hi])
        o = leaky(lat, tgt, v)
        out[key] = (c['total_cost'], float(np.sqrt((o**2).mean())))
    return out
if __name__ == '__main__':
    files = sorted(Path('data/SYNTHETIC').iterdir())[5000:5800]
    with Pool(12, initializer=init) as p: res = [r for r in p.map(job, files) if r]
    print(f'{len(res)} clean moving segments of 800')
    # counterexample: pid has the LOWEST drawn offset while having the HIGHEST cost
    cand = [r for r in res
            if r['pid'][1] == min(r[k][1] for k,_,_ in ARMS)
            and r['pid'][0] == max(r[k][0] for k,_,_ in ARMS)]
    print(f'segments where pid draws the tightest line AND costs the most: {len(cand)} '
          f'({100*len(cand)/len(res):.1f}%)')
    cand.sort(key=lambda r: (r['cnn_v4'][1]-r['pid'][1])/max(r['pid'][1],1e-6), reverse=True)
    print(f"\n{'seg':8} {'v':>5} {'peak':>6} " + ''.join(f"{k+' cost':>13}{k+' off':>10}" for k,_,_ in ARMS))
    for r in cand[:8]:
        print(f"{r['id']:8} {r['v']:5.1f} {r['peak']:6.2f} " +
              ''.join(f"{r[k][0]:13.1f}{r[k][1]:10.3f}" for k,_,_ in ARMS))
