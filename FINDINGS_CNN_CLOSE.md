# Informed close on the CNN line

Audit of every intervention tried against the learned controller, to decide whether the line is
exhausted or whether something is genuinely untested.

## First: today's headroom experiment was a replication

`7f3f2ff` ("Hard-example mining gives no gain", the ninth cnn negative, 2026-07-27) ran the same
experiment I ran today, with a warm start from `cnn_PM.pt`, matched pool, hyperparameters and
iteration count, differing only in whether batches were drawn uniformly or 50% from the re-scored
worst decile. Its findings, quoted:

> Mining does work as designed — best p99 (125.16), pool hard-set mean 114.2 → 102.8 — it just costs
> more in the bulk than it returns … median and p90 worsen while p99 improves.

Today's `hr500` reproduced that signature point for point: best p99 of any model tested (128.0), best
of the trained arms on the worst 100 (120.47), worse median on both holdouts, net null. The only
difference is the selection criterion — excess over the `31.24 + J*` floor rather than raw realised
cost, which is a genuine improvement in targeting — and it did not change the outcome.

The prior commit also states the arithmetic that predicts it: **76% of the cnn's cost sits outside
its worst decile**, so tail-targeted training cannot pay for itself. It adds the generalisation
"mining suits a tail-concentrated problem, which is `ff_pi`'s shape and not `cnn`'s" — which is
exactly what this session found independently from the other side, when `ff_pi_boot`'s 25 worst
segments turned out to carry 92% of its gap to the CNN.

I should have checked the record before spending the compute.

## The full list

Every cnn-directed intervention on record, with outcome:

| # | intervention | outcome |
|---|---|---|
| 1 | Online MPC on the neural plant (gradient, MPPI) | diverged / worse than steering zero |
| 2 | Shooting-teacher → distillation | negative (open-loop cannot reject drift) |
| 3 | Deterministic-only training | 56.38 — never learns drift rejection |
| 4 | Deterministic base → noisy fine-tune | superseded by soft-token BPTT |
| 5 | Rate-limited delta actions | diverged |
| 6 | Curvature + rate features | hurt (~1.3) |
| 7 | Past-history temporal branch | hurt |
| 8 | K-sample gradient averaging | hurt substantially |
| 9 | **Hard-example mining** | null (−0.052 ± 0.297) |
| 10 | **Capacity, 8× width (11k → 89k)** | null (+0.325 [−0.79, +1.85]) |
| 11 | **Preview length H=25 → 45** | null (+1.004 [−0.71, +2.57]) |
| 12 | **Gradient batch 32 → 96** | **real (−1.269 [−2.26, −0.61]) — but see below** |
| 13 | Measured gain prior (cfg `G`) | null (+0.25 [−0.10, +0.60]) |
| 14 | BC → PO (`dual_train.py`) | null vs matched control (−0.016) |
| 15 | Oracle distillation, three teacher constructions | three distinct failures |
| 16 | `steer_lookup` teacher + BC | teacher works, student fails |
| 17 | DAgger, BC+PO | null vs matched control |
| 18 | PO fine-tune from the BC student | initialisation is a net liability |
| 19 | c*-tracking objective | negative |
| 20 | Policy-proposed gradient-refined planning | worse |
| 21 | Ensemble of checkpoints | null on the mean |
| 22 | Terminal critic / actor-critic | ruled out by measurement |
| 23 | Difficulty specialisation (3-arm) | premise fails on 3000 hard segments |
| 24 | Feature selector / routing | dead, three independent lines |
| 25 | Disturbance observer | plant has no hidden state to observe |
| 26 | 2-DOF deterministic+stochastic transfer | premise holds, transfer fails |
| 27 | Warm-start from deterministic weights | worse than from scratch |
| 28 | Conditional integration | no headroom (clamp binds 1 step in 24,000) |
| 29 | Retrain on the 495 "weak" segments | **worse** than a matched control |
| 30 | Headroom-selected training (3-arm) | null — replicates #9 |
| 31 | Portfolio routing `cnn_v2`/`cnn_dual` | no measurable complementarity |

The pattern named in `0aeaaad` holds across all of them: **faster convergence, same ~48 ceiling.**

## The one thread that is genuinely open

Intervention #12, the gradient-batch arm (`ACC` 32 → 96), is the only non-null result on this list,
and its own commit message flags three unresolved problems:

1. **The mechanism is unproven.** `ACC=12` supplies 3× the data per iteration and the arms were
   matched on *iterations, not compute*, so the gain may be data rather than gradient-variance
   reduction. The confound was stated before the run, not after.
2. **It never converged.** The best checkpoint was the final iteration, still improving monotonically
   while the baseline oscillated. Real-sim selection over all 14 checkpoints picks the last one. The
   run was stopped mid-descent.
3. **It does not clearly beat the released `cnn_PM`.** It wins on `ALL[4000:4200]` (47.21 vs 48.20)
   and `ALL[5000:6000]` (48.48 vs 49.80) but loses on `ALL[:5000]` (48.27 vs 47.87).

So the honest status of the cnn line is *not* "everything was tried and everything failed". It is
"everything was tried and everything failed **except one arm that was never run to completion**".

## Recommended close

Close the line, with one exception. The defensible version of the remaining experiment is:

* rerun the gradient arm **matched on compute rather than iterations**, which resolves the
  data-vs-variance confound in the same run;
* train until real-sim score stops improving rather than to a fixed iteration count, since the last
  run was demonstrably stopped mid-descent;
* judge it against the released `cnn_PM` on all three splits, not just the two it won.

If that converges above ~47 on clean splits, the line reopens. If it flattens at ~48 like the other
eleven, the ceiling is a property of the method — policy optimisation through a chaotic stochastic
plant — and the remaining gap to the ~36 noise-free frontier needs a different method (MPC on a
smooth model, or value-based RL), which is what the README already concludes.

Everything else on the list is closed, and the two nulls added today (#29, #30) should be read as
confirmation rather than new information.
