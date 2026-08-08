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

# Chosen for driving character, and screened so the car is MOVING throughout (vmin > 5 m/s).
# 193 of 2000 pristine segments are stationary for part of the window -- the "cheapest" segment in
# the set is simply a parked car, which scores ~0.5 and shows nothing.
SEGMENTS = [
    ('06308', 'Gentle - straight cruise, 20 m/s'),
    ('05541', 'Typical - flowing curves, 19 m/s'),
    ('05913', 'Winding - busy steering, 26 m/s'),
    ('06585', 'Hard corner - 6.9 m/s2 at 32 m/s'),
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
