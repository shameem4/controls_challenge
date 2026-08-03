# Difficulty-scheduled gains: works on pid_boot, and a feedforward makes it unnecessary

Scheduling on DIFFICULTY rather than speed. Speed was the wrong variable: the measured plant gain
moves only 1.37 -> 1.83 across 5-34 m/s, and synthetic conditions at v=8/22/34 land within 17% of
each other.

## The difficulty signal: J*, exact and causal

The benchmark cost is a convex quadratic in the lataccel trajectory alone, so for any target window
the minimum achievable cost is the closed-form Tikhonov solve

    J*(tau) = min_c [5000*mean((c-tau)^2) + 100*mean((dc/dt)^2)],  (I + 2 D'D) c* = tau

O(n) by Thomas, no plant, no controller, no rollout -- and CAUSAL, since it runs over the 50-step
`future_plan` available at every step. It measures exactly the tracking-vs-jerk conflict: large only
when the target cannot be followed smoothly.

As a per-segment difficulty predictor of cnn_v2's cost (n=700):

    road features (6, linear)   R^2 0.293
    J* alone                    R^2 0.214
    J* + road features          R^2 0.361
    log J* -> log cost          R^2 0.493        <- multiplicative, use logs
    ceiling (noise-draw limit)  R^2 0.83

J* median is 3.00 against a median total cost of 45.07, so target difficulty is under 10% of a
typical segment's score -- the rest is the ~31 noise floor plus controller slack.

## Premise test on pid_boot: holds, with an asymmetric structure

Tuning separately on the easiest and hardest 300 of 1000 held-out segments (split by J*, medians
0.79 vs 12.51). Both prefer p=0.145; they differ in the INTEGRAL gain:

    p=0.145   ki:   0.050    0.075    0.100    0.140    0.200
    EASY           52.60    41.13    35.64    32.43   190.53    <- wants 0.14
    HARD          182.61   134.19   118.49   212.46  4506.67    <- wants 0.10, pays 79% at 0.14

Asymmetric: easy pays 10% at the hard optimum, hard pays 79% at the easy optimum. A fixed gain must
therefore sit at the hard optimum and forfeit the easy gain. That is what scheduling recovers.

## controllers/pid_fuzzy.py: two-rule Takagi-Sugeno on log J*

    mu   = sigmoid((log J*_preview - centre)/width)
    i(t) = i_easy + (i_hard - i_easy)*mu        (centre=1.0, width=0.8, set a priori, never swept)

Equal easy/hard gains reproduce pid_boot bit-for-bit.

    PRISTINE ALL[5000:6000]                  mean     median   better
      pid_boot nominal (0.195,0.100)        69.455    55.69      --
      retune fixed     (0.145,0.100)        70.533    56.57    262/1000   <- did NOT transfer
      fixed easy gains (0.145,0.140)        93.090    49.76    840/1000   <- median great, mean blows up
      FUZZY  i: 0.140 -> 0.100              66.833    53.31    745/1000

      fuzzy vs nominal          mean -2.622 [-5.81,+0.13]  median -1.743  745/1000
      fuzzy vs best fixed       mean -3.700 [-7.15,-0.74]  median -2.938  852/1000   <- CI excludes 0

    HEADLINE ALL[:5000]
      fuzzy 67.144 median 53.27 | nominal 68.412 median 55.49
      mean -1.268 [-2.67,+0.26]  median -1.7482  better 3633/5000  (32 sigma)

**Fails the pre-registered bar** (the benchmark is a mean and its CI includes zero on both splits),
but with the REVERSE signature of this project's usual false positive: median and sign test are
overwhelming and consistent across independent splits (medians -1.743 and -1.748), while only the
mean CI is wide. Two separate effects with different reliability:

    bucket      nominal    fuzzy    delta each   contribution to mean
    best 50%      35.00    33.31       -1.69          -0.845
    50-90%        68.46    69.06       +0.59          +0.238
    90-99%       140.98   137.40       -3.57          -0.321
    worst 1%    1188.39  1019.04     -169.35          -1.693

The median gain is solid (500 segments, 32 sigma). The mean gain is 65% driven by ten segments,
which is why the CI is wide. The 50-90% band is slightly worse -- the membership transition region,
where the gains are neither optimum.

Note the fixed retune p=0.145 won on BOTH in-sample subsets and LOST on the pristine split (+1.078).
Fixed-gain tuning overfit; the scheduler did not.

## Porting to ff_pi_boot: the premise does not hold

    ff_pi_boot (kp=0.1424, ki=0.1353, lead=2, gain_scale=1.79, i_clip=3.11, lam=4.74, boot=0.005)
      EASY: ki 0.080->41.33  0.105->32.20  0.1353->28.98  0.170->29.51  0.220->99.64
      HARD: ki 0.080->114.58 0.105->86.38  0.1353->82.94  0.170->99.82  0.220->329.74

**Both subsets peak at the SAME ki=0.1353 -- 0.00% available from scheduling.**

Why: ff_pi_boot has an inverse-plant FEEDFORWARD, which supplies the steady-state command directly,
so the integrator only trims a residual -- and the best gain for trimming a residual does not depend
on segment difficulty. pid_boot has no feedforward, so its integrator carries the whole command and
its optimal gain tracks the target's dynamics.

**The fuzzy scheduler was partially recovering what a feedforward supplies outright.** On a
controller that already has one there is nothing left to recover. Consistent with the ladder:
pid_boot 68.41 -> ff_pi_boot 51.22, where the feedforward is worth 17 points AND removes the
difficulty-dependence that made scheduling worth anything.

Incidentally the CMA value ki=0.1353, tuned on the whole distribution, is independently optimal on
both subsets -- a robust optimum rather than a compromise.
