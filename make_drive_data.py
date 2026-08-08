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
# The fifth is the counterexample, kept deliberately: a segment where `pid` really does draw the
# tightest line while costing the most. On 05772 it holds 0.083 m against `cnn_v4`'s 0.098 m, 15%
# tighter, while costing 39% more (69.4 vs 50.0).
#
# This slot previously held 06585, which was WRONG and actively misleading. That segment is
# corrupted: 27 of its target steps exceed the plant's physical rate limit, by up to 9.1x, with the
# target swinging +5.62 -> +1.33 -> -3.23 -> -5.64 in consecutive steps around t=19-21 s. About 80%
# of its ~8000 cost comes from roughly 2 s of data no controller could follow, and 7 of the 10
# highest-cost steps sit on those jumps. It is not a hard corner and was never a driving result.
# 0 of 495 sampled segments have more than 5 such steps; 06585 has 27. See `find_counterexample.py`.
#
# The honest counterexample is also a different REGIME than previously claimed. 05772 peaks at
# 0.70 m/s2 of lateral acceleration at 33 m/s -- gentle, fast, near-straight cruising, not a
# sustained corner. The earlier "persistent lag through a sustained corner" explanation does not
# survive. What survives is the mechanism: `pid`'s error is zero-mean chatter that the 3 s leaky
# integrator behind the drawn offset averages away and the cost charges in full. On clean data this
# is rare -- 2 of 637 segments, 0.3% -- which is itself the point: showing only this one would
# imply `pid` drives best, and the 205-segment sweep refutes that.
SEGMENTS = [
    ('05161', 'Gentle - light steering, 25 m/s'),
    ('05034', 'Typical - median segment, 25 m/s'),
    ('05150', 'Winding - busy steering, 27 m/s'),
    ('05115', 'Demanding - top decile activity, 25 m/s'),
    ('05772', 'Counterexample - pid draws the tightest line and costs 39% more'),
]

# Two families, so the comparison the visualisation exists to make is a controlled one: each
# trajectory controller shares a parent with a lataccel controller and differs ONLY in the error
# signal fed to the feedback term. `ff_pi_tau` is dropped -- it was visually identical to
# `ff_pi_boot` on 3 of the 4 old segments, so it cost a slot and showed nothing.
#
# Trajectory params are passed EXPLICITLY rather than relying on defaults. `pid_traj`'s defaults
# (k_y=0.60, k_psi=1.20) sit at a DC gain of 9.0 and diverge -- they are the pre-sweep values from
# `FINDINGS_TRAJ_PID.md`, kept only so the docstring's worked example matches. The values below are
# the tuned ones from that sweep.
CONTROLLERS = [
    # lataccel-error feedback
    ('pid',          'PID (lataccel error)',              'pid',        {}),
    ('ff_pi_boot',   'ff_pi_boot (lataccel error)',       'ff_pi_boot', dict(boot=0.005)),
    ('cnn_v4',       'cnn_v4 (learned, deliverable)',     'cnn',        dict(ckpt='cnn_v4.pt')),
    # trajectory-error feedback -- same parents, different error signal
    ('pid_traj',     'pid_traj (trajectory error)',       'pid_traj',
     dict(w=1.0, k_y=0.005, k_psi=0.96, tau=0.5, i=0.1)),
    ('ff_pi_traj',   'ff_pi_traj w=1 (trajectory error)', 'ff_pi_traj',
     dict(w=1.0, k_psi=0.16, tau=1.0)),
    ('ff_pi_traj25', 'ff_pi_traj w=0.25 (blended)',       'ff_pi_traj',
     dict(w=0.25, k_psi=0.40, tau=1.0)),
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
    # Plant constants, so the page can project where the CURRENT steer command is taking the car.
    # G(v) is the measured steady-state gain from steer to lataccel (`gain_fit.npy`, verified in
    # verify_plant.py); ALPHA is the per-step fraction of the way to that steady state, a
    # first-order stand-in for a plant whose real response has ~0.25 s of dead time.
    gain_fit = np.load('gain_fit.npy')
    data = dict(dt=DEL_T, gain_fit=[float(c) for c in gain_fit], alpha=0.40, segments=[])

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
