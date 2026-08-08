# Selective trajectory feedback: both arms null. Closing the line.

Follow-up to `FINDINGS_FF_TRAJ.md`, which left trajectory feedback at an exchange rate of 0.57
(jerk cost saved per unit tracking cost spent, break-even 1.0) and suggested the remaining lever was
to spend it *selectively* rather than uniformly. Two arms, both identity-gated.

## Arm A -- gate on local jerk share (`ff_pi_gate.py`)

Open the blend only where jerk dominates the local cost, measured causally with the cost's own
weights: `share = 10000*jerk_ema / (5000*track_ema + 10000*jerk_ema)`.

Tuning set `ALL[:60]` looked promising: 41.46 against the parent's 41.64. The gate is shut 98% of
steps (mean applied `w` = 0.0137), and a constant `w=0.10` costs +0.611, so linear scaling predicted
+0.084 at that dose against a measured -0.18 -- apparent selectivity.

**It did not reproduce.** Held out, `ALL[5000:5400]`, n=400, paired:

| arm | total | paired delta | 95% CI | better on | sign p |
|---|---|---|---|---|---|
| `ff_pi_boot` | **56.738** | -- | -- | -- | -- |
| gate lo=0.5 | 57.501 | +0.763 | [-0.095, +2.120] | 105/232 non-tied | 0.168 |
| gate lo=0.3 | 58.846 | +2.109 | **[+0.563, +3.905]** | 157/342 non-tied | 0.144 |

> **Corrected 2026-08-08 after a bug bash.** The numbers above are from the fixed implementation.
> The original run had `ff_pi_traj.update` delegating to its parent whenever `w == 0.0`, with the
> psi/y recursion placed after that early return -- so with the gate shut on ~97% of steps the
> trajectory state advanced on only **2.9%** of steps and was stale whenever the gate reopened. The
> arm was not testing the mechanism claimed. Fixed by making the recursion unconditional (bit-exact
> at `w=0` because `0.0*x` is exactly `0.0`), re-verified against all three identity gates.
> Correcting it made the arm **worse**, not better -- `lo=0.3` went from +0.719 to +2.109 and its CI
> now excludes zero. The pre-fix figures were: lo=0.5 +0.438 [-0.181, +1.300], lo=0.3 +0.719
> [-0.083, +1.661]. Conclusion unchanged in direction and strengthened.

The decisive detail is not the total, which is only weakly distinguishable, but the composition:
**jerk went UP** (+0.058 and +0.356). On the tuning set the gate bought jerk with tracking; on fresh
segments it buys nothing and merely spends tracking. The 60-segment result was run variance.

This is the falsification written into the controller's own docstring before running it:
`FINDINGS_SPECTRUM.md` located the cost gap in a narrow 0.5-1.0 Hz band, and that is a FREQUENCY
property. A local time-domain ratio has no reason to track it, and does not.

## Arm B -- per-step optimal blend of two controllers (`ff_pi_blend.py`)

Run `ff_pi_boot` and pure-trajectory `ff_pi_traj(w=1)` in parallel and pick the blend minimising the
predicted one-step cost. `u(b)` is affine in `b`, so `J(b) = 5000*e_pred^2 + 10000*du^2` is a scalar
quadratic with a closed-form minimum -- no search, one divide per step.

Null, and degenerately so. The optimiser is a bang-bang switch, not a graded chooser:

| b_hi | at lower bound | at upper bound | interior |
|---|---|---|---|
| 0.25 | 27.6% | 66.2% | **6.2%** |
| 1.0 | 46.8% | 27.2% | 26.0% |

Sitting at a bound 93.8% of the time at `b_hi=0.25`, it reproduces the constant-blend curve it was
meant to beat (44.93 vs 45.32 for a constant `w=0.25`), and is monotonically better as `b_hi -> 0`.
The one-step model -- a static gain and a lag fraction on a plant with 0.25 s dead time -- is too
crude to rank interior blends, so it always sees one endpoint as better.

## One measurement worth keeping

Across 11,590 steps, the jerk share of local cost has **median 7.8%, mean 10.9%**, exceeding 50% on
2.0% of steps. Tracking dominates almost everywhere. That is the plainest available statement of why
trajectory feedback cannot pay here regardless of how it is scheduled: there is very little
jerk-dominated territory to spend it on.

## Closing the line

Four arms now, converging: `pid_traj` (6.3x worse), `ff_pi_traj` (0.57 exchange rate, crossover at
1.75x current jerk weight), `ff_pi_gate` (selectivity fails out of sample), `ff_pi_blend`
(degenerate). The two structural reasons from `FINDINGS_TRAJ_PID.md` still explain all of it:

1. Two extra integrators on a dead-time-dominated plant, where phase margin already binds.
2. No new information -- the dataset has no road, so trajectory error is reconstructed by
   integrating the same lataccel signal the controller already sees.

Recommend closing unless the cost function changes. The one condition that revives it is a jerk
weight >= 1.75x the current 2x-per-squared-unit, at which `ff_pi_traj(w=0.10)` crosses over.
Nothing promoted; `master` untouched.
