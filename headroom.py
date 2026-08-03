"""Rank training segments by REAL HEADROOM: cost minus the achievable floor.

The floor has two parts (see FINDINGS_FLOOR.md, FINDINGS_FFPI_HARD.md):
  * J*  -- the closed-form Tikhonov optimum on that segment's target trajectory, causal and exact
  * 31.24 -- the causal noise floor, irreducible for any controller

A segment with cost 200 and J* 190 has no headroom. A segment with cost 200 and J* 3 does. Ranking
by raw cost confuses the two, and ranking by which of two tied networks happened to win ranks by
noise -- that is what the 495-segment experiment did, and it trained the model backwards.

Usage: python headroom.py <ckpt> <start> <n> [out.csv]
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map
from tinyphysics import (TinyPhysicsModel, TinyPhysicsSimulator,
                         CONTROL_START_IDX, COST_END_IDX)
from controllers.pid_fuzzy import jstar

NOISE_FLOOR = 31.24
_M = [None]
def model():
    if _M[0] is None:
        _M[0] = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    return _M[0]


def one(f, ckpt):
    from controllers.cnn import Controller
    f = str(f)
    c = TinyPhysicsSimulator(model(), f, controller=Controller(ckpt=ckpt), debug=False).rollout()
    tau = pd.read_csv(f)['targetLateralAcceleration'].values[CONTROL_START_IDX:COST_END_IDX]
    js = jstar(tau)
    return dict(seg=f, total=c['total_cost'], lat=c['lataccel_cost'] * 50.0, jerk=c['jerk_cost'],
                jstar=js, floor=NOISE_FLOOR + js, excess=c['total_cost'] - (NOISE_FLOOR + js))


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'cnn_dual.pt'
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    out = sys.argv[4] if len(sys.argv) > 4 else f'headroom_{start}_{n}.csv'
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    S = ALL[start:start + n]
    D = pd.DataFrame(process_map(partial(one, ckpt=ckpt), S, max_workers=24, chunksize=2, disable=True))
    D.to_csv(out, index=False)
    print(f'=== {ckpt} on ALL[{start}:{start+n}] ===', flush=True)
    print(f'  cost   mean {D.total.mean():8.3f}  median {D.total.median():6.2f}', flush=True)
    print(f'  floor  mean {D.floor.mean():8.3f}  median {D.floor.median():6.2f}', flush=True)
    print(f'  excess mean {D.excess.mean():8.3f}  median {D.excess.median():6.2f}   '
          f'at/below floor: {(D.excess <= 0).sum()}/{len(D)}', flush=True)
    E = D.sort_values('excess', ascending=False).reset_index(drop=True)
    tot = D.excess.clip(lower=0).sum()
    for k in (100, 250, 500, 1000):
        print(f'  top {k:4} by headroom hold {100*E.excess[:k].clip(lower=0).sum()/tot:5.1f}% of positive excess   '
              f'(mean excess {E.excess[:k].mean():7.2f}, mean J* {E.jstar[:k].mean():6.2f})', flush=True)
    print(f'  wrote {out}', flush=True)


if __name__ == '__main__':
    main()
