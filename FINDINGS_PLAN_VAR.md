# Planning the next 20 actions for mean AND variance: the objective is right, the planner is not

Follows from treating the control action as a continuation of the token/action history rather than a
scalar knob (`FINDINGS_ENDOGENOUS_NOISE.md`). Two consequences were worth building on: the plant's
output variance is controllable, and the action enters a 20-step context so it should be planned as a
coherent continuation rather than as independent scalars.

## The objective was derived, not guessed

The expected cost decomposes exactly:

    E[(c - tau)^2]       = (mu - tau)^2 + sigma^2
    E[(c_k - c_{k-1})^2] = (mu_k - mu_{k-1})^2 + sigma_k^2 + sigma_{k-1}^2

Every controller in this project has optimised only the first half of each and treated `sigma^2` as a
constant to be dropped. It is not constant and it is not exogenous, so the second half is a real term
with a real gradient.

The planner: horizon 20 (the plant's context length -- actions beyond it cannot influence the current
emission), plan parameterised by 3 coefficients on a smooth polynomial basis so every candidate is a
coherent continuation, rolled forward in `expected` mode (deterministic and differentiable, so none of
the sampling chaos that killed the earlier MPC), Adam on the coefficients, first action emitted,
re-plan each step.

## The variance term does exactly what it claims

| planner variant | E[Var] | vs WVAR=0 |
|---|---|---|
| constant-hold base, WVAR=0 | 0.005251 | — |
| constant-hold base, WVAR=1 | 0.003624 | **-31%** |
| policy-rollout base, WVAR=0 | 0.002219 | — |
| policy-rollout base, WVAR=1 | 0.001481 | **-33%** |

Two independent planner configurations, the same answer: scoring the variance reduces the plant noise
the controller actually experiences by about a third. The mechanism is real and it is usable.

## The planner is anti-productive, and the diagnostic is unambiguous

32 pristine segments, torch sim:

| arm | track | jerk | total | E[Var] |
|---|---|---|---|---|
| `cnn_v4` | 27.57 | 22.73 | **50.30** | **0.001066** |
| planner, constant-hold base, KSTEP=4 | 38.33 | 24.00 | 62.33 | 0.001182 |
| planner, constant-hold base, KSTEP=12 | 122.17 | 58.36 | 180.54 | 0.003624 |
| planner, policy base, KSTEP=6, WVAR=0 | 103.17 | 44.33 | 147.50 | 0.002219 |
| planner, policy base, KSTEP=6, WVAR=1 | 120.71 | 38.31 | 159.02 | 0.001481 |

**`theta = 0` is exactly `cnn_v4`'s action.** The planner therefore starts at the baseline, and every
gradient step on the predicted cost makes the true closed loop worse -- monotonically: 4 steps gives
62, 12 steps gives 181. The predicted objective is anti-correlated with the realised one. That is
certainty equivalence failing: optimising the mean trajectory ignores that the realised trajectory
diverges from it, and corrections that look optimal against the mean are harmful under sampling.

Replacing the naive constant-hold base plan with the policy rolled forward in expected mode helped
substantially (208.74 -> 147.50) and did not change the verdict.

This is the fourth independent model-based-planning failure here, after gradient MPC and MPPI on the
neural plant, sequential linearisation (`FINDINGS_SEQLIN.md`), and -- externally -- Ryan Lei's CEM-MPC
and nurikserikbayev's OSQP/ARX at ~57. The consistent cause is that a model-based objective on this
plant does not predict realised cost.

## The result that closes it

Look at the variance column. `cnn_v4` realises **0.001066**, lower than the explicit
variance-optimising planner ever reaches (0.001481). A policy trained end to end on the true sampled
cost already keeps the plant quieter than a planner that optimises quietness directly.

That is the same shape as every other finding in this line: the mechanism is real, and the end-to-end
policy has already found it without being told. It also explains why the variance term, though
correctly derived and demonstrably effective inside a bad planner, has nothing left to give a good
one.

## Status

Rejected as a controller. Kept as the derivation and measurement of a term the project had been
silently dropping, and as the reason not to attempt model-based planning here a fifth time.

---

## Addendum: the planner discarded its plan every step

The planner re-solves from scratch at every control step -- `theta` is reset to zero, optimised for
KSTEP gradient steps, the first action is emitted, and the whole plan is thrown away. Standard MPC
carries the plan forward; omitting that was a defect.

Two hypotheses for why it would matter, one right and one wrong:

* **convergence** -- a handful of gradient steps from zero never reaches the optimum. Correct: warm
  starting is worth **31 points**, 159.02 -> 127.86.
* **action jitter** -- consecutive actions coming from different partially-converged optima would be
  rough, and `FINDINGS_ENDOGENOUS_NOISE.md` showed roughness raises the plant's own variance ~30%, so
  the planner would be inflating the noise it was minimising. **Refuted by measurement.**

| arm | total | E[Var] | mean abs du |
|---|---|---|---|
| `cnn_v4` | 50.30 | 0.001066 | 0.01378 |
| planner, no warm start | 159.02 | 0.001481 | 0.02188 |
| planner, warm-started | 127.86 | 0.002419 | 0.02263 |

Warm starting left the action roughness unchanged (0.02188 -> 0.02263) and raised the variance. So the
planner's actions are ~1.6x rougher than the policy's either way, and the gain came from convergence
alone. The roughness originates in the BASE plan -- recomputed from a new true state each step -- not
in the correction on top of it.

The verdict is unchanged: 127.86 against a baseline of 50.30, from a planner whose `theta = 0` is
exactly that baseline.
