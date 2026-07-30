# Cynic review / bug bash

Adversarial pass over this session's claims. Everything below was re-run, not recalled.

## 1. CONFIRMED BUG — corrected

**`grad_probe.py` used a mismatched saturation mask.** It took the mask from a `sample`-mode rollout
of the cnn policy, then differentiated a *separate* `gumbel`-mode rollout with soft tokens. Those
consume RNG differently, follow different trajectories, and saturate at different steps — so the
"gradient at saturated steps" was measured at steps where a *different* rollout had saturated.

Corrected (`grad_probe2.py`, mask taken from the rollout actually differentiated):

```
                saturated steps   |grad| SAT    normal    ratio
st_clamp=False        4/9576       1.505e+02   1.067e+01   14.11
st_clamp=True         4/9576       1.577e+02   1.143e+01   13.79
```

* The MAIN conclusion survives and strengthens: gradient at saturated steps is **14x larger** than
  normal, definitively not zero. The structural argument holds independently — an action enters the
  plant's 20-step input window, so the clamp kills one gradient path out of twenty-one.
* The SECONDARY claim is **wrong**. The commit says `st_clamp` "roughly doubles the gradient at
  saturated steps (67 -> 121)". It does not: **+5%** (150.5 -> 157.7), ratio unchanged. The fix is
  even less consequential than reported.
* Caveat on both: only 4 saturated steps in 9576 (0.04%), so these means are over 4 samples.

## 2. OVERCLAIM — "the floor is ~29"

The floor is `irreducible jerk (11.33) + lat floor at the effective feedback lag`:

```
 lag k   lat floor   TOTAL
     3      17.04    28.37     <- what I quoted, using the ONSET lag
     4      22.71    34.04
     5      28.39    39.72     <- the BULK/peak response lag
```

I quoted **~29**, which assumes the effective lag is 3 — the *onset* of the impulse response. But the
same measurement puts the bulk and peak at **5 steps**, which would give 39.7. I picked the
optimistic end and presented it as the estimate.

There is a consistency check that constrains it: the honest frontier (~36) is an *achieved* score,
and a floor cannot exceed an achieved score. So effective lag <= 4 and **the floor lies in 29-34**.
That check also means my "frontier sits 7 points above the floor, mutual corroboration" was too
tidy — the frontier is doing the work of bounding the floor, not independently confirming it.

Consequence: cnn's headroom is **13-19 points**, not "~19".

## 3. UNRESOLVED — dualcnn (46.89), the unpromoted result

The result is verified (two splits, CI excluding zero, sign test 11.6 SD, controlled against a
from-scratch arm on identical pool and budget). The *mechanism* is not, and the leading hypothesis
is refuted:

```
                        loss    |grad|    |w|     after 20 opt steps
cnn_PM (released)       41.90   217.37   8.628         45.63
dualcnn AFTER BC        40.71   172.62   8.608         46.52
```

Weight norms indistinguishable; gradients *smaller* at the BC clone, not larger; fixed optimisation
buys *less*. Every prediction of the flat-vs-sharp-minimum story fails. Combined with round 2 not
compounding, the honest position is: this checkpoint is better, the procedure is n=1 and
unexplained. **Not promoted.**

## 4. UNRESOLVED — the gradient-batch arm

`ACC=12` beat its from-scratch control by -1.269 (CI excluding zero, 595/1000), but:
* confounded with **3x compute** — arms matched on iterations, not compute. Flagged before the run,
  never resolved.
* it beat cnn_PM on two clean splits and **lost** on `ALL[:5000]` (48.27 vs 47.87).
* it never converged (best checkpoint was the final iteration).

Not a usable result in its current state.

## 5. VERIFIED SOUND

* **Master deliverables** all reproduce exactly (pid 100.3010, ff_pi 69.2125, ff_pi_tuned 74.8741,
  cnn 75.4548); `cnn` byte-identical to the `v1-learned-47.87` tag; `ff_pi_rl2(hold=0)` bit-identical
  to `ff_pi_tuned`.
* **ff_pi_rl2 (52.30)** — the saturation detector has **zero false positives in 79,800 steps** (max
  non-clamped |dlat| = 0.4888 against a 0.495 threshold; 70 steps hit the clamp exactly). Bootstrap
  CI, distribution-free sign test, replicated on 15,000 never-evaluated segments.
* **pid_look (110.76 -> 84.12)** — `k0=0` bit-identical to stock PID, stock reproduces its quoted
  110.76 exactly, 4537/5000 segments improved.

## 6. RECURRING METHODOLOGICAL FAULTS

1. **Worst-N subset sweeps are biased.** Selecting segments by baseline cost then comparing on them
   flatters any perturbation (regression to the mean). Produced at least two false positives.
2. **torch-sim validation disagrees with the real sim in DIRECTION** (seen on the preview arm). Only
   ONNX-sim numbers should be quoted.
3. **t-tests on heavy-tailed deltas overstate significance.** Use bootstrap CI + sign test.
4. **My mechanism explanations have a poor record: four of five proposed were later refuted** —
   zero-gradient clamp, detuning shape, tail "instability", and the BC flat-minimum story. Results
   held up far better than explanations. Treat any causal story here as unverified unless it names
   the measurement that tested it.

## 7. WHAT SHOULD CHANGE

* `FINDINGS_CLAMP.md` and its commit overstate the `st_clamp` effect (2x -> actually 5%). Corrected
  here; the file should carry a pointer.
* `FINDINGS_TAIL.md` §15 should read "floor 29-34" and drop the mutual-corroboration framing.
* README on master is unaffected — none of these claims reached it.
