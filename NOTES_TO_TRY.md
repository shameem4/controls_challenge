# To try — ideas borrowed from other solutions

From a review of public comma.ai controls_challenge solutions (2026-08-06). Ranked by expected value.
Nothing here has been run; these are candidates, not results.

## 1. Sequential linearisation for the per-segment optimiser  (highest value)

`nurikserikbayev/comma-controls-challenge` optimises each segment by iteratively linearising: refine
the action sequence, measure the model's misprediction, re-optimise with the correction applied, then
polish with random search. Reported **29.37** total.

Our `steer_opt.py` uses coordinate descent and converges at **38.25** on 128 pristine segments. Theirs
is a materially tighter optimum.

**Why it matters, and what it is NOT for.** This is a measuring instrument, not a controller — a
segment-fingerprinting replay is the seed exploit and is not submittable. But `FINDINGS_SEED_ORACLE.md`
established that our per-segment difficulty measure is a loose upper bound, and that the additive floor
(`sigma^2*26614 + J*`) is not a valid bound at all. A tighter optimum sharpens the one question we
could not answer cleanly: **how much of the remaining excess is actually recoverable**. Every targeting
experiment so far (mining, headroom selection, band analysis) has been aimed using a floor we now know
is wrong.

## 2. Independent system ID disagrees with ours by ~25%

Their excitation-based ARX identification puts the plant DC gain at **~2.0**. Ours is
`G(v) = 0.0093v + 1.34` (about 1.5-1.7 at typical speeds), and `ff_pi` then applies a tuned
`gain_scale = 1.79` on top, landing near 2.7 effective.

Two independent identifications disagreeing by this much is worth resolving. If their 2.0 is right,
our `gain_scale` is absorbing a systematic ID error rather than a genuine detuning, which would explain
why `gain_scale` needed tuning at all and why the tau-gated detune in `ff_pi_tau` helps.

## 3. Quadratic-in-speed feedforward  (cheap)

`karti-ai` uses `steer = k1*net + k2*net*v + k3*net*v^2` (net = target lataccel - roll), fit to the
SIMULATOR's measured response rather than to real steering logs. Our `G(v)` is linear in v.

Test whether a `v^2` term buys anything. ~20 minutes as a grid. Low expected value -- our
`FINDINGS_SYSID.md` found gain depends on operating point rather than speed alone, so a richer
*speed* polynomial may be fitting the wrong variable.

## Explicitly NOT worth doing

**Online MPC.** Their OSQP implementation -- proper QP, 50-step preview, on an identified ARX model --
scores **~57**, behind our `ff_pi_tau` (49.47). Combined with our own neural-plant MPC failures
(`FINDINGS_ORACLE.md`: the plant is chaotic, so candidate rankings are noise), that is two independent
implementations landing behind the classical controller. Direction closed.

**Their 8-step target average** for "zero-lag smoothing" is a weaker version of our Tikhonov solve,
which is the analytic cost optimum by construction.

## Calibration note

comma.ai's own blog (`blog.comma.ai/rlcontrols`) reports **48.0** from CMA-ES over 6 parameters of a
custom feedback controller, and that PPO "improves over training but doesn't converge to good values".
Both match our experience -- our classical line sits at 49.47 and our own RL attempts needed soft-token
BPTT to work at all. Treat their 48.0 with some caution: it was tuned on **20 segments**, and our
`ff_pi_tuned` produced a 16% phantom gain from fitting on 60.

---

# Neuro-fuzzy repos reviewed (2026-08-06) — nothing to borrow

Searched "neuro fuzzy logic driving github" and reviewed the substantive hits.

| repo | what it is | verdict |
|---|---|---|
| `nickgkan/neuro-fuzzy-vehicle-controller` | inputs: distance-to-obstacle, angle-to-goal; output: velocity; 3- and 5-rule bases | different problem |
| `exarchou/Fuzzy-Systems` | coursework: DC-motor fuzzy control, vehicle fuzzy regulation, neuro-fuzzy for regression/classification | no control detail, no numbers |
| `AlinaBaber/...AUV...` | Mamdani self-tuning PID + NN gain classifier | reviewed separately; kp range [250,251] makes the P-path tuning a 0.4% no-op, `Kp=max(range)` makes the NN a constant lookup, no online learning despite the framing, no reported results |
| `sadegh-msm/fuzzy-driver`, `hinsonan/FuzzyLogicSelfDrivingTruck` | sensor-based obstacle avoidance | different problem |

**Why none of it transfers.** The published "fuzzy/neuro-fuzzy driving" work is overwhelmingly REACTIVE
NAVIGATION -- pick a speed or a heading to avoid obstacles, two inputs, a handful of rules. Our problem
is REFERENCE TRACKING: follow a specified lataccel trajectory under a quadratic cost with a jerk term,
through a dead-time-dominated (L/T ~ 1.7) stochastic plant whose disturbance is essentially white
(|autocorrelation| <= 0.011). Fuzzy control is well suited to the former -- no model needed, smooth
blending of common-sense rules -- and has little to offer the latter, where the cost function is known
analytically and its optimum (the Tikhonov solve) can be computed in closed form.

That is consistent with what `FINDINGS_FUZZY_FLC.md` measured directly: the best member of our TS fuzzy
family is its LINEAR member, i.e. the PI we already ship.

**One signal on the remaining untested variant.** ANFIS-style learning of the fuzzy parameters is the
open item from `FINDINGS_FUZZY_FLC.md` (optimise all 25 consequents rather than shaping them with one
GAMMA). `nickgkan` reports that their learned neuro-fuzzy system UNDERPERFORMS the hand-tuned fuzzy
system on seen maps while generalising better to unseen ones -- learned rules losing to tuned rules
in-distribution. Combined with our own BC result (teacher works, student loses; the bar is R^2 > 0.99
in action space) and a GAMMA sweep that peaked exactly at linear, the prior on that variant is low.
