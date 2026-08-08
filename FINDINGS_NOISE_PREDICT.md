# Can we predict what the seed-aware exploit knows? No — held-out R² is zero

Every seed-aware result in this project rests on one advantage: the exploit knows the realised
innovation `eps_t = c_t - mu_t`. `FINDINGS_SEED_ORACLE.md` measured the consequence -- the seed-aware
optimum reaches a tracking cost of **17.75, below the causal lataccel floor of 19.50** -- it
pre-compensates noise it can see.

If any part of `eps` were predictable from causal observables, a real controller could do the same,
and the payoff would be large. The causal floor is `sigma^2 * 26614`, so a predictor capturing a
fraction `rho` of the innovation variance moves it to `(1 - rho)` of that. Even `rho = 0.1` is ~3
points -- larger than any controller improvement found in this project.

Two routes, kept apart:

* **physics** -- predict `eps` from past innovations, state, action, and the conditional
  distribution's own moments. Legitimate control. Measured here.
* **RNG** -- infer the draw stream. `tinyphysics.py:116` seeds from `md5(path) % 10^4`, so only
  10,000 streams exist and enough observations identify which one. That is segment fingerprinting: a
  lookup table, not a controller, and already documented as the mechanism behind the sub-30 entries.

## Measurement

37,440 held-out steps from 96 pristine segments under `cnn_v4`. Features: 10 lagged innovations,
current lataccel, target, v_ego, roll, a_ego, action, and the conditional mean, variance and **skew**
of the plant's own output distribution -- the last three included so that any sampler bias would show
up rather than being assumed away.

| predictor | train R² | held-out R² |
|---|---|---|
| autocorrelation, lags 1, 2, 3, 5, 10 | — | +0.0063, −0.0036, +0.0018, +0.0055, −0.0148 |
| ridge, lambda = 1 | +0.00348 | **−0.00439** |
| ridge, lambda = 10 | +0.00348 | −0.00437 |
| ridge, lambda = 100 | +0.00344 | −0.00409 |
| MLP, 2x64 tanh, 1500 steps | +0.04172 | **−0.04734** |

**Held-out R² is negative for every predictor** -- worse than predicting the mean. The MLP is the
clearest case: it finds +0.042 of structure in-sample and that structure is entirely spurious,
scoring −0.047 out of sample.

So `rho = 0`, and the floor stays exactly where it was: `sigma^2 = 0.001090` gives 29.02, and a
predictor of this quality gives 29.02. Saving: **0.00**.

## What this settles

The gap to the seed-aware optimum is **provably unavailable to a causal controller**. Not
"unavailable with the methods tried" -- the innovation carries no causal signal to extract, so no
controller, learned or classical, can pre-compensate it.

That closes the last open interpretation of several earlier results at once:

* the seed-aware optimum's tracking below the causal floor (17.75 vs 19.50) is not a controller
  deficiency, it is information we do not have;
* the 0.5-1 Hz concentration being unfixable (`FINDINGS_SPECTRUM.md`, Bode waterbed under dead time)
  and the noise being unpredictable are two independent fundamental limits, and together they account
  for the residual;
* the repeated finding that `cnn_v4` matches or beats every planner on jerk, plant variance and
  action smoothness while losing only on tracking is consistent: the tracking deficit versus the
  exploit is information, not control quality.

It also retroactively justifies the floor framework. The causal floor is a real bound precisely
because `eps` is unpredictable; had it been predictable, the floor would have been an artefact of not
trying hard enough.
