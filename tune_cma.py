"""CMA-ES tuning of the classical controllers.

Motivation: comma's own RL-controls post reports reaching cost 48.0 on this challenge by
optimising just *6 parameters* with CMA-ES, in under 10 minutes -- while their PPO failed to
converge. Our classical controllers were only ever coarse GRID-searched, which sits on lattice
points and misses parameter interactions. This asks whether ff_pi's 59 was a real ceiling or
just under-tuning.

The sim is seeded per segment (md5 of the path), so re-evaluating the same parameters on the
same segments is deterministic -- the objective is noise-free and CMA-ES sees a clean landscape.

Usage: python tune_cma.py <pid|ff_pi> [iters] [nseg]
"""
import sys, os, numpy as np, importlib, cma
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

ALL = sorted(Path('data/SYNTHETIC').iterdir())
TUNE = ALL[2000:2000 + int(os.environ.get('NSEG', 60))]     # disjoint from every eval split
VAL = ALL[500:620]                                           # clean held-out
_M = [None]


def m():
    if _M[0] is None:
        _M[0] = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    return _M[0]


# name, lower, upper, is_int
SPACE = {
    'pid_tuned': [('p', 0.0, 1.0, False), ('i', 0.0, 0.6, False), ('d', -0.5, 0.5, False)],
    'ff_pi': [('kp', 0.0, 1.0, False), ('ki', 0.0, 0.5, False), ('lead', 0.0, 12.0, True),
              ('gain_scale', 0.4, 3.0, False), ('i_clip', 0.5, 20.0, False),
              ('lam', 0.1, 12.0, False)],
}


def decode(kind, x):
    """Map the unbounded CMA vector through a sigmoid into the box, rounding ints."""
    out = {}
    for xi, (name, lo, hi, is_int) in zip(x, SPACE[kind]):
        v = lo + (hi - lo) / (1.0 + np.exp(-xi))
        out[name] = int(round(v)) if is_int else float(v)
    return out


def encode(kind, params):
    """Inverse of decode(): the CMA vector that reproduces a given parameter dict.
    Used to START the search from the shipped defaults rather than the box midpoint --
    otherwise CMA begins far from a known-good point and wastes its budget."""
    x = []
    for name, lo, hi, _ in SPACE[kind]:
        v = float(np.clip(params[name], lo + 1e-6, hi - 1e-6))
        frac = (v - lo) / (hi - lo)
        x.append(float(-np.log(1.0 / frac - 1.0)))
    return x


def run(f, kind, params):
    C = importlib.import_module(f'controllers.{kind}').Controller
    c = C(**params)
    return TinyPhysicsSimulator(m(), str(f), controller=c, debug=False).rollout()['total_cost']


_WARNED = [False]


def ev(kind, params, files, workers=12):
    """Evaluate a parameter set. Failures are reported once and then penalised -- an earlier
    version swallowed them silently, so every candidate scored 1e6 and the 'tuning' was a no-op."""
    try:
        r = process_map(partial(run, kind=kind, params=params), files,
                        max_workers=workers, chunksize=max(1, len(files) // workers), disable=True)
        v = float(np.mean(r))
        if not np.isfinite(v):
            if not _WARNED[0]:
                print(f'  !! non-finite cost for {params}'); _WARNED[0] = True
            return 1e6
        return v
    except Exception as e:
        if not _WARNED[0]:
            print(f'  !! evaluation FAILED for {params}: {type(e).__name__}: {e}'); _WARNED[0] = True
        return 1e6


def main(kind, iters, nseg):
    files = ALL[2000:2000 + nseg]
    base = importlib.import_module(f'controllers.{kind}').Controller()
    defaults = {n: float(getattr(base, {'lam': 'lam', 'lead': 'lead'}.get(n, n))) for n, *_ in SPACE[kind]}
    print(f"CMA-ES tuning {kind}: {len(SPACE[kind])} params, {len(files)} tune segs, {iters} iters")
    b_tune = ev(kind, {}, files)
    print(f"  baseline (shipped defaults): tune={b_tune:.2f}  val={ev(kind, {}, VAL):.2f}", flush=True)

    print(f"  starting from defaults: {defaults}", flush=True)
    es = cma.CMAEvolutionStrategy(encode(kind, defaults), 0.6,
                                  {'popsize': 10, 'maxiter': iters, 'verbose': -9, 'seed': 1})
    best = (b_tune, {})
    it = 0
    while not es.stop():
        xs = es.ask()
        costs = [ev(kind, decode(kind, x), files) for x in xs]
        es.tell(xs, costs)
        it += 1
        i = int(np.argmin(costs))
        if costs[i] < best[0]:
            best = (costs[i], decode(kind, xs[i]))
            print(f"  it{it:3d} tune={costs[i]:7.2f}  {best[1]}", flush=True)
        elif it % 5 == 0:
            print(f"  it{it:3d} (best so far {best[0]:.2f})", flush=True)

    print(f"\nBEST on tune set: {best[0]:.2f}   params={best[1]}")
    v_new = ev(kind, best[1], VAL)
    v_old = ev(kind, {}, VAL)
    print(f"clean held-out [500:620]:  defaults={v_old:.2f}   tuned={v_new:.2f}   "
          f"delta={v_new - v_old:+.2f}")
    if v_new > v_old:
        print("  -> tuned is WORSE on held-out: the gain was tune-set overfitting")


if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'ff_pi'
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    nseg = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    main(kind, iters, nseg)
