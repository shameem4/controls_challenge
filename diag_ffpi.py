"""Where does ff_pi_boot actually fail, and is that failure the controller's fault?

"Difficult" is defined as EXCESS over the segment's own achievable floor, not as raw cost. The floor
is the closed-form Tikhonov optimum J* over the true cost window (steps 100-500), which is the best
any causal-or-not controller could score on that target trajectory ignoring plant noise. A segment
with cost 200 and J* 190 is not difficult -- it is expensive and there is nothing to win. A segment
with cost 200 and J* 3 is a controller failure.

Instruments per segment: cost split into tracking vs jerk, the plant's MAX_ACC_DELTA rate clamp
saturation, the integrator, and the road/target statistics, so the failures can be characterised
rather than just counted.

Usage: python diag_ffpi.py [start] [n]
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map
from tinyphysics import (TinyPhysicsModel, TinyPhysicsSimulator, MAX_ACC_DELTA,
                         CONTROL_START_IDX, COST_END_IDX, DEL_T)
from controllers.ff_pi_boot import Controller as Base
from controllers.pid_fuzzy import jstar

_M = [None]
def model():
    if _M[0] is None:
        _M[0] = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    return _M[0]


class Probe(Base):
    """ff_pi_boot, instrumented. Behaviour is untouched -- it only records."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.sat = 0; self.maxi = 0.0; self.plat = None; self.n = 0; self.satrun = 0; self.maxrun = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        if self.plat is not None and abs(current_lataccel - self.plat) >= 0.99 * MAX_ACC_DELTA:
            self.sat += 1; self.satrun += 1; self.maxrun = max(self.maxrun, self.satrun)
        else:
            self.satrun = 0
        self.plat = current_lataccel
        self.n += 1
        u = super().update(target_lataccel, current_lataccel, state, future_plan)
        self.maxi = max(self.maxi, abs(float(self.integ)))
        return u


def one(f):
    f = str(f)
    c = Probe(boot=0.005)
    sim = TinyPhysicsSimulator(model(), f, controller=c, debug=False)
    cost = sim.rollout()
    d = pd.read_csv(f)
    tau = d['targetLateralAcceleration'].values[CONTROL_START_IDX:COST_END_IDX]
    v = d['vEgo'].values[CONTROL_START_IDX:COST_END_IDX]
    lat = np.array(sim.current_lataccel_history)
    dtau = np.diff(tau) / DEL_T
    return dict(total=cost['total_cost'], lat=cost['lataccel_cost'] * 50.0, jerk=cost['jerk_cost'],
                jstar=jstar(tau), sat=c.sat, maxrun=c.maxrun, maxi=c.maxi,
                absmax=float(np.abs(tau).max()), absmean=float(np.abs(tau).mean()),
                dtau_max=float(np.abs(dtau).max()), dtau_rms=float(np.sqrt((dtau**2).mean())),
                v_mean=float(v.mean()), v_min=float(v.min()),
                lat_max=float(np.abs(lat).max()))


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    S = ALL[start:start + n]
    R = process_map(one, S, max_workers=24, chunksize=2, disable=True)
    D = pd.DataFrame(R); D['seg'] = [str(x) for x in S]
    D['excess'] = D.total - D.jstar
    D.to_csv(f'diag_ffpi_{start}_{n}.csv', index=False)

    print(f'=== ff_pi_boot on ALL[{start}:{start+n}]  mean {D.total.mean():.3f}  median {D.total.median():.2f} ===\n', flush=True)
    print(f'  cost split: tracking {D.lat.mean():.2f} ({100*D.lat.mean()/D.total.mean():.0f}%)   '
          f'jerk {D.jerk.mean():.2f} ({100*D.jerk.mean()/D.total.mean():.0f}%)', flush=True)
    print(f'  achievable floor J*: mean {D.jstar.mean():.2f}  median {D.jstar.median():.2f}', flush=True)
    print(f'  EXCESS over floor:   mean {D.excess.mean():.2f}  median {D.excess.median():.2f}\n', flush=True)

    D = D.sort_values('excess', ascending=False).reset_index(drop=True)
    for k in (10, 25, 50, 100):
        sh = D.excess[:k].sum() / D.excess.sum() * 100
        print(f'  worst {k:4} by excess hold {sh:5.1f}% of total excess', flush=True)

    print(f'\n  {"group":14} {"n":>5} {"total":>9} {"J*":>8} {"excess":>9} {"jerk%":>6} '
          f'{"sat":>6} {"maxrun":>7} {"max|i|":>7} {"|tau|max":>8} {"dtau_rms":>8} {"v":>6}', flush=True)
    groups = [('worst 25', D[:25]), ('26-100', D[25:100]), ('101-300', D[100:300]), ('rest', D[300:])]
    for nm, g in groups:
        print(f'  {nm:14} {len(g):5} {g.total.mean():9.2f} {g.jstar.mean():8.2f} {g.excess.mean():9.2f} '
              f'{100*g.jerk.mean()/g.total.mean():6.1f} {g.sat.mean():6.1f} {g.maxrun.mean():7.1f} '
              f'{g.maxi.mean():7.2f} {g.absmax.mean():8.2f} {g.dtau_rms.mean():8.2f} {g.v_mean.mean():6.1f}', flush=True)

    print('\n  correlation of log(excess) with candidate difficulty signals:', flush=True)
    le = np.log(np.clip(D.excess.values, 1e-3, None))
    for c in ['jstar', 'absmax', 'absmean', 'dtau_max', 'dtau_rms', 'v_mean', 'v_min', 'sat', 'maxi', 'lat_max']:
        x = D[c].values
        x = np.log(np.clip(x, 1e-3, None)) if c in ('jstar', 'dtau_max', 'dtau_rms', 'absmax', 'absmean') else x
        print(f'    {c:10} r = {np.corrcoef(le, x)[0,1]:+.3f}', flush=True)

    hi = D[:100]
    print(f'\n  worst-100 vs rest:  saturating segments {100*(hi.sat>0).mean():.0f}% vs '
          f'{100*(D[100:].sat>0).mean():.0f}%     |tau|max {hi.absmax.mean():.2f} vs {D[100:].absmax.mean():.2f}', flush=True)
    print(f'  worst-100 jerk share {100*hi.jerk.mean()/hi.total.mean():.0f}% vs rest '
          f'{100*D[100:].jerk.mean()/D[100:].total.mean():.0f}%', flush=True)


if __name__ == '__main__':
    main()
