# Planning on sampled rollouts: certainty equivalence really was costing ~13 points

`FINDINGS_PLAN_VAR.md` isolated the expected-mode planner's failure to tracking alone -- every
smoothness measure was already at or better than the trained policy. The suspected cause was certainty
equivalence: optimising `J(mean trajectory)` rather than `E[J(trajectory)]`. Since the cost is
quadratic, `E[J] = J(mean) + curvature * variance`, so the two genuinely differ.

The chaos result does not forbid sampling. `FINDINGS_SEQLIN.md` measured +3.67 mean cost from
perturbing actions by 1e-4 -- but that is against ONE FIXED noise realisation, where a token flip is a
discontinuity. `E[cost | plan]` averages over realisations and is smooth in the plan; the difficulty
becomes estimator variance, which is what MPPI and common random numbers address.

## Setup

MPPI over the same 3 smooth basis coefficients (a 3-dimensional search, small enough to cover with a
handful of candidates), K candidates per step each scored on M sampled rollouts, **common random
numbers** across candidates so comparisons are paired, exponential-weighted update, warm-started
across steps, plus the plan-consistency filter BETA. Candidate 0 is always the incumbent, so an update
can never be worse than standing still.

## Results, 16 pristine segments

| config | track | jerk | total | E[Var] | mean abs du |
|---|---|---|---|---|---|
| `cnn_v4` baseline | **25.08** | 19.20 | **44.28** | 0.001052 | 0.01258 |
| MPPI sampled, BETA=0 | 130.67 | 38.02 | 168.69 | 0.001777 | 0.02560 |
| MPPI sampled, BETA=0.4 | 63.05 | 22.24 | 85.29 | 0.001156 | 0.01653 |
| MPPI sampled, BETA=0.8, K=12 M=4 | 42.18 | 17.32 | 59.51 | 0.001024 | 0.01054 |
| MPPI sampled, BETA=0.8, K=16 M=12, 3 iters | 39.70 | **16.67** | **56.37** | **0.001019** | 0.01004 |
| MPPI **expected**, BETA=0.8, K=12 | 54.25 | 17.94 | 72.19 | 0.001061 | 0.01008 |
| theta frozen at 0, BETA=0.8 (no search) | 91.90 | 17.13 | 109.03 | 0.001022 | 0.00930 |

## What it establishes

**Sampling beats certainty equivalence by 12.7 points** at matched settings (59.51 vs 72.19), and the
entire gain is in tracking (54.25 -> 42.18, -22%) -- exactly the term that had been isolated. The
diagnosis was right and the fix works.

**Search and plan-smoothing are both necessary and interact strongly.** Search without smoothing:
168.69. Smoothing without search: 109.03. Together: 56-59. Neither is optional, and neither alone is
close.

Note the theta=0 arm is *not* "cnn_v4 plus smoothing" -- it is a degenerate planner whose base plan is
the policy rolled forward and then heavily filtered, so it lags. It measures what the filter does
without anything re-optimising against it, which is why it looks bad.

**Compute helps, with steep diminishing returns.** Roughly 4.5x the planning budget (K 12->16, M 4->12,
2->3 iterations) buys 3.1 points, 59.51 -> 56.37. Closing the remaining 12 points at that exchange rate
is not reachable.

**Every non-tracking metric is better than the trained policy**, again: jerk 16.67 vs 19.20, plant
variance 0.001019 vs 0.001052, action roughness 0.01004 vs 0.01258. The planner produces a smoother,
quieter controller than `cnn_v4` and simply aims worse -- 39.70 vs 25.08.

## Verdict

Best planner 56.37 against a 44.28 baseline. Rejected as a controller, but this closes the question
properly rather than by assumption: model-based planning here is not blocked by chaos, and its main
identified defect was real and fixable. What remains is that a 3-coefficient plan re-optimised over a
20-step horizon against a sampled model simply aims less well than a policy trained end to end on the
true cost -- and the gap costs more compute to close than it is worth.

This is the fifth model-based planning attempt in the project and the first whose failure is fully
attributed: not chaos, not roughness, not noise, not convergence, not plan consistency, not certainty
equivalence. Just aim.

---

## Addendum: plan parameterisation is not the limit

The last untested lever was the plan family -- 3 smooth coefficients over 20 steps is restrictive, and
a richer parameterisation might aim better. Tested with a piecewise-constant **block** basis, which
scales cleanly from n=1 (one number held over the horizon) to n=20 (fully independent per-step
actions). A polynomial basis cannot answer this: past degree ~4 the `k**j` columns are collinear.

Sampled MPPI, K=12, M=4, 2 iterations, BETA=0.8, 16 pristine segments:

| resolution | track | jerk | total | E[Var] | mean abs du |
|---|---|---|---|---|---|
| n=1 | 74.39 | 17.26 | 91.65 | 0.001012 | 0.01100 |
| n=3 | 49.09 | 16.71 | 65.81 | 0.001018 | 0.01046 |
| **n=6** | 43.39 | 16.55 | **59.94** | 0.001009 | 0.01053 |
| n=10 | 44.79 | 17.60 | 62.38 | 0.001041 | 0.01055 |
| n=20 (per-step) | 49.49 | 18.88 | 68.36 | 0.001075 | 0.01149 |
| poly n=3 (reference) | 42.18 | 17.32 | 59.51 | 0.001024 | 0.01054 |

**There is an interior optimum at n ~ 6**, and it lands exactly where the 3-coefficient polynomial
already was (59.94 vs 59.51 at the same budget). Resolution was never the binding constraint.

Past the optimum the cost rises and so do both action roughness (0.01053 -> 0.01149) and plant variance
(0.001009 -> 0.001075). Two causes, both expected: 12 candidates cannot cover a 10-20 dimensional
search, and a rougher plan reintroduces the endogenous noise penalty of
`FINDINGS_ENDOGENOUS_NOISE.md`. Full per-step freedom is the worst setting tested that still has a
working search.

So the best planner across everything tried remains **56.37** (poly n=3, K=16, M=12, 3 iterations)
against `cnn_v4`'s **44.28**. Every lever has now been swept -- objective (mean vs expected cost),
rollout mode (expected vs sampled), plan consistency, warm starting, compute, and parameterisation --
and the residual is still aim.
