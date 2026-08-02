# Oracle distillation: the teacher is the hard part

Goal: exploit the fixed per-segment seed OFFLINE to build near-optimal action sequences, then distil
them into a causal net. The student would read only observations, so it is a legitimate controller --
distillation from a privileged teacher, not fingerprinting. Three teacher constructions were tried
and all three failed, each for a different and instructive reason.

## Why the seed is exploitable at all

`tinyphysics.py:116` seeds the RNG from `md5(filename)`, so each segment has ONE noise realization
forever. Verified: open-loop replay of a recorded action sequence reproduces the closed-loop cost to
`0.00e+00`, while the same actions against a different draw cost 6x more (43.03 -> 238.47).

What is fixed is the STREAM OF DRAWS, not an additive shock sequence -- a draw maps to a token
through the CURRENT distribution, so changing actions changes probs and the same draw yields a
different outcome. The exploit needs only determinism, which holds.

## Attempt 1: gradient through the recursion -- diverges

Warm-started from cnn_v2's own actions (43.624 on 16 segments), Adam at lr 2e-4 with gradient
clipping: not one iterate out of 50 beat the warm start; cost sat at 220-229. The recursion is
chaotic, so a 400-step gradient is noise. Reproduces the project's earlier online-MPC failure (9353).

Two bugs found en route, both worth remembering:
  - `mode='sample'` is `bins[multinomial(...)]` -- a hard index lookup with ZERO gradient. Optimizing
    through it silently returns the warm start unchanged. Use `mode='gumbel'` + soft_tokens.
  - patching `pol.__call__` on an INSTANCE does nothing; Python resolves dunders on the class. The
    observation capture collected nothing and only surfaced as an empty-list crash.

## Attempt 2: sequential greedy probing -- myopic

`oracle_build.py`. Snapshot history AND cuda RNG, try K candidates, roll each forward HZ steps,
restore exactly, commit the best. Verified exact: identical actions after restore reproduce
bit-for-bit (max|diff| 0.0), different actions diverge. Derivative-free, so chaos is irrelevant.

    K=9 HZ=6 SPAN=0.30, 32 segments:  ORACLE 84.875 (lataccel 38.2, jerk 46.7)  vs cnn_v2 43.624

Worse than the net it was meant to teach. A first version scored 90.567 because `seg_cost` charged
jerk only WITHIN the probe window, leaving the transition into the window free -- so each step picked
independently and the chatter became jerk. Fixing that recovered only 90.6 -> 84.9. The remaining
problem is myopia: the impulse response spans ~5 steps, so each greedy step partially undoes the
previous commitment, and the probe assumes the base controller continues smoothly when in fact the
oracle deviates again at the next step.

## Attempt 3: greedy tracking of the closed-form optimal trajectory -- worse

The benchmark cost is a convex quadratic in the LATACCEL TRAJECTORY alone, independent of the plant,
so its minimiser is the Tikhonov solve `(I + lam D'D) c* = tau`, lam = W_jerk/W_track = 2. We compute
its cost analytically at **6.69** (lataccel 1.53 + jerk 5.16); an independent implementation
(RyanL2/commacontrol) reports **6.880** with the rate limit and 1024-bin quantisation imposed. Two
derivations, same number.

Judging candidates purely on `(realised lataccel - c*)^2`:

    K=13 HZ=6 SPAN=0.40, 32 segments:  ORACLE 182.731 (lataccel 51.7, jerk 131.1)

**c\* is not an achievable target under noise.** It is optimal as a trajectory you could impose
directly, but forcing the plant onto it means fighting every shock, and the correction costs far more
jerk than the tracking saves. This is the same reason `ff_pi` runs detuned and why every good
controller here deliberately lags. A useful negative: "track the analytic optimum" is the wrong
objective for a causal controller.

## What the exploit bands actually are

Earlier framing conflated two different things. Corrected:

| band | what it does |
|---|---|
| ~7 | Bypasses the plant -- injects the closed-form optimal LATACCEL trajectory directly. Not control. |
| ~20-30 | Exploits the fixed seed -- optimises ACTIONS offline against a known realisation. |
| ~31 | Causal bound with the plant in the loop (FINDINGS_FLOOR.md). |
| ~36-40 | Honest frontier. RyanL2's honest entry: 39.9, against our 46.91. |

## Where this stands

Building a good oracle is itself an open problem on this plant. The principled route is a per-segment
**min-plus DP over the 1024 output bins** with the true cost -- exact, non-myopic, immune to chaos
(RyanL2 used exactly that to verify against the evaluator to delta 0). That is a real build.

And the distillation obstacle is unchanged: MSE learns `E[u_oracle | obs]`, and the component that
makes the oracle good is a function of the realised draws, which are white (|autocorr| <= 0.011), so
it averages toward zero. The student would inherit the nominal policy -- measured at 54.43 sampled,
worse than cnn_v2's 46.26. That measurement should gate any training run: on a WORKING oracle,
compute how much of `u_oracle` is predictable from observations and whether cnn_v2 already emits it.

---

## Calibration against a documented reference entry (RyanL2/commacontrol)

Its README lists every arm with scores, which pins the bands far better than leaderboard
descriptions do:

| controller | score | nature |
|---|---|---|
| continuous_lookup_noclip | 6.880 | injects the closed-form optimum into the simulator |
| continuous_lookup | 6.89 | same, respecting rate limits |
| token_lookup | 7.05 | replays a DP-optimal output-token sequence |
| steer_lookup | 39.9 | per-segment coordinate descent on real steering commands (a LOOKUP) |
| cem_mpc | ~76 | honest online CEM-MPC |
| PID | ~68 | upstream baseline |

Three corrections and one confirmation.

**Our earlier band framing was wrong.** This file previously said ~20-30 was "seed-exploited action
optimisation". It is not. Careful per-segment coordinate descent on ACTIONS tops out at **39.9**.
Everything below ~10 is TRAJECTORY INJECTION, which bypasses the plant entirely. There is essentially
nothing in between -- the gap is not populated because optimising actions against a known seed is
genuinely hard, not because nobody tried.

**That corroborates the three failed oracle attempts above.** A careful coordinate descent reaches
39.9, only ~7 points better than our fully CAUSAL 46.91. So the payoff from seed exploitation via
actions is small, and the difficulty is intrinsic rather than a defect of our search.

**Their honest controller is far behind ours.** `cem_mpc` at ~76 is the only causal entry there;
`steer_lookup` is a per-segment lookup despite the "genuine steering commands" framing. Our cnn_v2 at
46.91 beats their honest method by ~29 points.

**Confirmation:** their closed-form optimum 6.880 (with rate limit + 1024-bin quantisation) matches
our independently derived Tikhonov optimum of 6.69 (lataccel 1.53 + jerk 5.16, lam = 2). Two
implementations, same quantity.

### Consequence

Nothing cheap remains to borrow. The closed-form c* we already compute in `ff_pi.smooth()`. Reaching
39.9 requires exactly the per-segment oracle construction that failed three ways above. And the
honest-frontier estimate of ~36 should be treated with more suspicion now: the only documented honest
entry we can actually inspect scores 76, and 39.9 -- which we had been reading as near-frontier -- is
a lookup.

---

# The distillation plan, completed: the teacher is worse than the student

Plan: segment -> closed-form optimal trajectory -> ideal control actions -> BC a CNN on
(observation, ideal action) pairs. Built and verified end to end. It fails for a structural reason,
not an engineering one.

## Building the teacher (ideal_mpc.py)

The optimal TRAJECTORY is closed form -- the cost is a convex quadratic in the lataccel trajectory
alone, so `(I + 2 D'D) c* = tau`. Getting ACTIONS from it took four attempts:

    one-step inversion        116679   H[1]=0.02, so hitting c*[t+1] needs ~50x gain -> rails
    open-loop deconvolution      365   jerk 18.3 (good!) but lataccel 347 -- nothing corrects drift
    DMC without free response  19000   omits past action changes still propagating; jerk 2388
    DMC, complete                 51.9 correct

The free-response term was the whole difference (19000 -> 62 -> 51.9 after tuning MU). The increment
model assumes the plant is settled with u held at u(t-1); it is not, and without
`free_j = sum_m (SSTEP[j+1+m] - SSTEP[m]) * du(t-m)` the solver keeps re-commanding motion already
on its way. Reachability weighting (rows scaled by SSTEP[j+1]) matters too, since the first ~3
horizon steps are nearly unreachable and an unweighted solve over-drives to reach them.

    MU sweep (HH=25, 16 segments):  0.02 -> 1281   0.1 -> 196   0.5 -> 62.6
                                    1.0 -> 52.1    2.0 -> 51.9  5.0 -> 54.8   12.0 -> 55.5

## Why the plan cannot work

    DMC teacher   51.9        cnn_v2   43.6   (same split, same seed, torch sim)

**The teacher is 8 points worse than the student it was meant to teach.** Distilling it produces a
student bounded by ~52.

That is not a tuning shortfall -- 51.9 is the classical plateau. Four independent model-based designs
now agree: ff_pi 59.06, ff_pi_boot 51.22, this DMC 51.9, RyanL2's CEM-MPC ~76. Model-based control on
this plant tops out at ~51-52, and the learned net at 46.9 is already past it.

So a useful teacher must beat 46.9, and the only way to do that is PRIVILEGED information -- knowing
the realised draws. But those draws are white (|autocorr| <= 0.011), so MSE distillation learns
E[u_oracle | obs] and the privileged component averages to zero; the student inherits the nominal
policy, measured at 54.43.

**The teacher must beat the student to be worth distilling, and building a controller better than
cnn_v2 is the original open problem.** The pipeline is sound and cannot bootstrap past its own input.

## Byproduct worth keeping

`ideal_mpc.py` is a working linear-MPC controller on this plant (51.9 torch / classical-plateau
class), built from the analytic reference plus a correct DMC formulation. It is the first MPC in this
project that works at all -- earlier online-MPC attempts diverged (9353) or landed at 944-16237. It
does not beat ff_pi_boot, but it independently confirms where the model-based ceiling is.
