# Sequential linearisation does not transfer: the objective is a step function of the actions

Ported from `nurikserikbayev/comma-controls-challenge` (reports 29.37) to sharpen our per-segment
difficulty instrument, which `steer_opt.py` leaves at 38.25. It does not work here, and the reason is
measurable and general.

## The implementation was sound

The benchmark cost is exactly quadratic in the lataccel trajectory, and the trajectory is locally
linear in the actions, so around any plan the full 400-step problem has a closed form:

    c ~= c0 + H (u - u0)
    (A H'H + B H'D'D H + lam I) d = -(A H'(c0 - tau) + B H'D'D c0)

with `H` a lower-triangular Toeplitz of the impulse response, `lam` a Levenberg-Marquardt trust
region, and the system factorised once per `lam` since `H` is shared across segments. Every candidate
is replayed through the true plant with the RNG reset and accepted per segment only on true cost.

One real bug was found and fixed on the way. Measured as a one-step impulse over 12 steps, the kernel
never decayed -- flat at ~0.37, summing to 4.25 against a verified DC gain of 2.4 -- because the plant
is autoregressive on its own lataccel output and a short probe has not settled. The truncated kernel
implied every action has a permanent effect. Measuring a **sustained** offset and differencing it
gives a kernel that decays properly and sums to 2.124, reconciling with `verify_plant.py`.

Even with the correct kernel the optimiser stalls: 3 accepted steps on iteration 1, 1 on iteration 2,
then nothing. 50.30 -> 50.23, against `steer_opt`'s 50.30 -> 38.25.

## Why: the objective is not locally smooth at any scale

Perturbing all 400 actions of a converged plan by a uniform random `eps` (64 segments):

| eps | mean cost | delta | segments changed |
|---|---|---|---|
| 0.0001 | 54.00 | **+3.67** | 23/64 |
| 0.001 | 138.87 | +88.53 | 64/64 |
| 0.003 | 269.66 | +219.33 | 64/64 |
| 0.01 | 447.23 | +396.89 | 64/64 |
| 0.1 | 750.02 | +699.69 | 64/64 |

A single action nudged by 0.01 at t=200 changes 39/64 segments and costs +20.1.

**There is no usable trust region.** At `eps = 1e-4` the step is already destructive; below that the
sampled trajectory is bit-identical, so there is no signal at all. The plant emits a discrete token,
and a change in the action changes which bin the fixed uniform draw selects; that flips the trajectory
onto a different realisation, which then cascades through the autoregression. The cost is a step
function of the actions with chaotic jumps, not a smooth surface with curvature to exploit.

This is the same property that killed MPC on this plant (`README`: "the plant is chaotic: tiny action
perturbations produce divergent predicted trajectories, so the expected-cost RANKING of candidate
action sequences is noise... it's a bad-landscape problem, not a bad-gradient one"). Sequential
linearisation is a gradient method in disguise and inherits exactly that failure.

## What this says about what the optimiser is actually doing

It reframes `steer_opt`. Coordinate descent is not descending a smooth surface -- it is **searching
over discrete token outcomes**, proposing changes and keeping the ones that happen to land on a
luckier noise realisation. That is why it needs true-cost acceptance at every level, why it took 8
annealed sweeps to reach 38.25, and why the reference implementation pairs its linearisation with a
random-search polish rather than relying on it.

So the route to a sharper difficulty measure is *more and better stochastic search* -- larger
populations, annealing, CEM over action perturbations -- not better gradients. Filed as such.

## Status of the borrowed-ideas list

* sequential linearisation -- **rejected**, this document
* DC gain disagreement -- **resolved**, `FINDINGS_GAIN_MODEL.md`: their ~2.0 was closer than our
  1.42-1.69, our `gain_scale` was absorbing the error, and the speed slope does not exist
* noise magnitude disagreement -- **resolved**, `verify_plant.py`: definition mismatch, our figure stands
* quadratic-in-speed feedforward -- now argued against by the gain finding: the measured gain is flat
  in speed, so a richer *speed* polynomial fits a variable that does not matter
