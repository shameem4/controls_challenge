"""Generate trajectory data for the interactive driving visualisation.

The dataset has no road geometry, but lateral acceleration and speed determine curvature exactly:

    kappa   = a_lat / v^2          heading rate  psi_dot = v * kappa = a_lat / v

So integrating the TARGET lataccel reconstructs the intended path, and integrating the ACHIEVED
lataccel reconstructs where the car actually went. The lateral drift between them IS the tracking
error, drawn to scale rather than illustrated.

Runs the real numpy simulator so every number matches the benchmark. Records per step: target and
achieved lataccel, the steer command, speed, road roll, and the two cost components, for each
controller on each segment.

Usage: python make_drive_data.py [out.json]
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from tinyphysics import (TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX,
                         COST_END_IDX, DEL_T, LAT_ACCEL_COST_MULTIPLIER)

# Screened so the car is MOVING throughout (vmin > 5 m/s): 193 of 2000 pristine segments are
# stationary for part of the window, and the "cheapest" one is simply a parked car.
#
# The earlier hand-picked set was chosen for driving character alone, and turned out to be
# unrepresentative in a way that actively misled: its most dramatic segment was one of the ~30%
# where `pid` shows the least lane deviation despite costing far more. These five are drawn from a
# 205-segment sweep (ALL[5000:5240], `segsel.py`) by binning on steering activity and taking the
# median-cost segment of each band, so the character spread is deliberate but the ranking within
# each is typical. On the first four, `cnn_v4` has both the lowest cost AND the lowest drawn lane
# deviation, which is the population behaviour (it wins cost on 145/205 and path on 125/205).
#
# The fifth is the counterexample, kept deliberately. On 06585 `pid` really does hold the tightest
# line -- 0.425 m drawn deviation against `cnn_v4`'s 0.655 m -- while costing 52% more (7967 vs
# 5244). It is not a contradiction: `pid`'s error is zero-mean chatter, which the 3 s leaky
# integrator behind the drawn offset averages away and which the cost charges for in full, while
# the others hold a persistent lag through the sustained corner. The eye lowpasses; the cost does
# not. Showing only the first four would hide a real effect; showing only this one, as the earlier
# set effectively did, implies `pid` drives best, which the 205-segment sweep refutes.
SEGMENTS = [
    ('05161', 'Gentle - light steering, 25 m/s'),
    ('05034', 'Typical - median segment, 25 m/s'),
    ('05150', 'Winding - busy steering, 27 m/s'),
    ('05115', 'Demanding - top decile activity, 25 m/s'),
    ('06585', 'Counterexample - pid holds the tightest line and costs 52% more'),
]

CONTROLLERS = [
    ('pid',        'PID (stock baseline)',            'pid',        {}),
    ('ff_pi_boot', 'ff_pi_boot (feedforward + PI)',   'ff_pi_boot', dict(boot=0.005)),
    ('ff_pi_tau',  'ff_pi_tau (best classical)',      'ff_pi_tau',  {}),
    ('cnn_v4',     'cnn_v4 (learned, deliverable)',   'cnn',        dict(ckpt='cnn_v4.pt')),
]


def run(model, path, mod, kw):
    import importlib
    C = importlib.import_module('controllers.' + mod).Controller
    sim = TinyPhysicsSimulator(model, str(path), controller=C(**kw), debug=False)
    cost = sim.rollout()
    lo, hi = CONTROL_START_IDX, COST_END_IDX
    return dict(
        lat=[round(float(x), 4) for x in sim.current_lataccel_history[lo:hi]],
        steer=[round(float(x), 4) for x in sim.action_history[lo:hi]],
        cost=round(float(cost['total_cost']), 2),
        lat_cost=round(float(cost['lataccel_cost'] * LAT_ACCEL_COST_MULTIPLIER), 2),
        jerk_cost=round(float(cost['jerk_cost']), 2),
    )


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'drive_data.json'
    model = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    root = Path('data/SYNTHETIC')
    lo, hi = CONTROL_START_IDX, COST_END_IDX
    data = dict(dt=DEL_T, segments=[])

    for seg, label in SEGMENTS:
        f = root / f'{seg}.csv'
        if not f.exists():
            print(f'  skip {seg} (missing)', flush=True)
            continue
        df = pd.read_csv(f)
        entry = dict(
            id=seg, label=label,
            target=[round(float(x), 4) for x in df['targetLateralAcceleration'].values[lo:hi]],
            v=[round(float(x), 3) for x in df['vEgo'].values[lo:hi]],
            roll=[round(float(np.sin(x) * 9.81), 4) for x in df['roll'].values[lo:hi]],
            runs={},
        )
        for key, name, mod, kw in CONTROLLERS:
            entry['runs'][key] = run(model, f, mod, kw)
            print(f'  {seg} {key:12} cost {entry["runs"][key]["cost"]:9.2f}', flush=True)
        data['segments'].append(entry)

    data['controllers'] = [dict(key=k, name=n) for k, n, _, _ in CONTROLLERS]
    Path(out).write_text(json.dumps(data, separators=(',', ':')))
    print(f'  wrote {out}  ({Path(out).stat().st_size / 1024:.0f} KB)', flush=True)


if __name__ == '__main__':
    main()
