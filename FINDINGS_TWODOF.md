# 2-DOF: deterministic nominal net + stochastic corrector

*Recorded on branch `cnn-optim`, last updated 2026-08-01. Headline numbers quoted here are those current at the time; `master` has since moved on — see the README for the figures that stand today.*

Proposal: train net A on the deterministic plant, net B on the realistic plant, combine. Unlike
every other second-network idea tried here, **the premise held** -- and the transfer still failed.

## The premise is real: nominal and rejection have different optima

`det_train.py` trains AblNet cfg PM on the expected-value plant, same recipe as cnn_v2 otherwise
(TRAIN=ALL[2000:4000], TBPTT=60, 1000 iters, Adam 2e-4), selecting checkpoints on the DETERMINISTIC
plant since that is the question being asked.

On ALL[4200:4520], same split for both:

    net              mode        lataccel     jerk    total
    det (A)          expected       11.05     6.00    17.04
    cnn_v2           expected       17.20     7.74    24.95
    det (A)          sample         36.58    17.85    54.43
    cnn_v2           sample         26.84    19.42    46.26

A is **32% better at nominal tracking**, with jerk 6.00 near the analytic Tikhonov floor of 5.16.
It is 8.17 worse as an actual controller. Near-perfect mirror image.

### Two stale numbers corrected

The README claimed a "~36 noise-free ceiling" from an old deterministic-only run. It was stale in
BOTH directions: every current net beats it without training there (cnn_v2 24.95, cnn_PMG 25.56,
cnn_PM 25.84), and a net that targets it reaches **17.04**. It was undertraining, not a ceiling.

Separately, this run reproduces the README's old 56.38 sampled figure almost exactly (val_sample
54-58, converging on 56.4) from an independent run with a better recipe -- so that number was never
an artifact; it is the stable price of never learning drift rejection.

## How A fails: placidly, not brittlely

Under noise A has WORSE lataccel (36.58 vs 26.84) but BETTER jerk (17.85 vs 19.42). Brittle policies
thrash; this one stays smooth and drifts off target. That is the signature of absent integral action.

`controllers/det_pi.py` bolts a plain classical PI on the frozen net -- no training, two gains:

    bare A            54.83
    ki=0.03           50.86   <- best
    ki=0.02           51.53
    kp=0.1  ki=0      57.11
    kp=0.2  ki=0      82.68
    kp=0.3  ki=0.05  751.71

Every proportional gain hurts, badly. Only the slow standing correction helps, exactly as the
placid-drift diagnosis predicts. But it recovers only **46%** of the gap (3.97 of 8.57). So about
half of A's deficit is missing rejection, and half is A behaving off-distribution once noise moves
the state off the noise-free manifold it trained on.

## The trained corrector: fails the bar

`twodof_train.py` / `controllers/twodof.py`: u = A(frozen) + B(trained on the stochastic plant).
B zero-initialised so training starts at exactly bare A -- verified, B=0 scores 56.436 against
det_train's 56.44 for the same checkpoint, so the harness passes A through unchanged. A stays frozen
by design; fine-tuning it would make this an initialisation scheme, and those were shown worthless
here (BC->PO was entirely run variance against a matched control).

    surrogate val (best)            44.169
    real-sim selection ALL[4000:4200]  46.909  (twodof_0450)

    TEST pristine ALL[4200:5000], n=800
      twodof (A+B)  48.869   median 45.23
      cnn_v2        50.720   median 44.50
      mean -1.851 [-6.37,+0.71]  median +0.5890  better 296/800

**FAILS.** CI includes zero, the median is +0.59, and only 37% of segments improve -- the typical
segment is worse and a few tail wins carry the mean. Fourth instance of this exact pattern today
(pid_pend, dualcnn checkpoints, ensemble, this).

The surrogate over-rated it again: 44.169 surrogate vs 48.869 real sim. Third surrogate/real-sim
divergence today, after the gain prior and the MoE gate.

## What this establishes

A genuinely better nominal tracker exists, and its advantage does not transfer through either a
classical corrector (46% recovered) or a trained one (fails on median and sign test). The obstacle
is not the corrector's capacity -- it is that A's nominal skill is defined on a state distribution
the closed-loop system does not visit. A is optimal for a plant that does not exist: the
expected-value plant is BIASED, not merely noise-free (a PID scores 29 expected vs 68 sampled).

The remaining untested variant is joint fine-tuning of A and B, which is explicitly an
initialisation scheme and therefore already answered: worth zero here against a matched control.

---

## Variant 2: warm-start a fresh CNN from the deterministic weights -- also negative

The frozen 2-DOF failed because A's nominal skill is defined on a state distribution the closed loop
never visits. Warm-starting removes exactly that constraint: A is free to move onto the real
distribution while keeping whatever nominal skill transfers. It is the natural next variant and the
one the failure analysis points at.

Prior evidence was genuinely mixed. The README records "deterministic base -> noisy fine-tune"
at 49.70 against 49.96 from scratch -- a -0.26 that we now know sits well inside the ~1.0
run-variance band, so it was a NULL recorded as a marginal win. But that transferred a ~36-nominal
net; ours is 17.04, and it predates soft-token BPTT.

Run: `TAG=warmdet SEED=0 TBPTT=60 TRAIN_N=2000 ablate.py PM 1000 gumbel_soft ckpts/det_0700.pt`.
The matched control already existed -- cnn_v2 IS the from-scratch SEED=0 1000-iteration run of the
same recipe, so initialisation is the only difference. Checkpoints selected on ALL[1200:1440], the
same protocol cnn_v2 used.

    selection trace: 56.07 (it0, bare A) -> 47.37 (it175) -> flat through it450
                     47.4 47.9 47.4 48.5 ... 46.9 47.2 49.5 48.9 47.2 47.0 47.5
    BEST ablwarmdet_0300 = 46.902

    TEST pristine ALL[4200:5000], n=800
      warmdet  51.512  median 45.83
      cnn_v2   50.720  median 44.50
      mean +0.792 [-0.30,+1.86]  median +0.5939  better 256/800

**Worse than from scratch** on mean and median, with only 32% of segments improving.

Budget caveat, stated and then discounted: training died at iteration 450 of 1000, so warmdet got
450 stochastic iterations against the control's 1000 (though 1450 total including the deterministic
phase). The selection trace is flat for 275 iterations with no trend, so finishing would very
likely land in the same band. The caveat is real but weak.

## The deterministic-pretraining direction is closed

Three variants, all negative, and together they say something specific:

    frozen A + classical PI       50.86  (recovers 46% of A's deficit; every kp>0 hurts)
    frozen A + trained B          48.87  vs cnn_v2 50.72 -- CI spans zero, median +0.59, 37% better
    warm-start A, fine-tune all   51.51  vs cnn_v2 50.72 -- worse on mean AND median, 32% better

A's nominal advantage (17.04 vs 24.95, a real 32% edge) does not transfer in ANY form: not frozen
with a classical corrector, not frozen with a learned one, not as an initialisation. The consistent
explanation is that the expected-value plant is BIASED, not merely noise-free (a PID scores 29
expected vs 68 sampled), so A is optimal for a plant that does not exist. Its skill is not
partially transferable -- it is defined on the wrong object.

The remaining variant, joint fine-tuning of A and B together, is an initialisation scheme by another
name and is answered by both the BC->PO null and this one.
