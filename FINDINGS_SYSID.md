# Classical system identification: open-loop step responses across operating conditions

A designed experiment (`sysid.py`), not observational fitting. Conditions are SYNTHESISED -- constant
v_ego, roll, a_ego, operating lataccel -- so each is an independent variable rather than a correlated
property of whichever segments got picked. The plant samples, so each row of a batch is an
independent noise realisation and a batch gives the ensemble mean response directly.

## Two protocol bugs found first, both of which produced physics-looking nonsense

**Non-equilibrium operating point.** The first version set `c0` (starting lataccel) and `u0` (steer)
independently. Steady state satisfies `c_ss = roll + G(v)*u0`, so unless they agree the plant is
mid-transient and the step response rides on an ongoing drift. That yielded DC gains of 0.397-2.933
and "overshoot" of 136%, 163%, 253% -- impossible for an open-loop low-pass plant, which is what
prompted the check.

**The null test that caught it.** Rerun each condition with du=0; anything that moves is drift, not
dynamics:

    u0=-1.0   pre=-0.7833   drift over 40 steps = -0.4041   <- NOT settled
    u0= 0.0   pre=-0.0025   drift = -0.0058                 <- settled
    u0=+1.0   pre=+1.5886   drift = -0.0074                 <- settled

Fixed by pinning `u0 = (c0 - roll)/G(v)`, raising SETTLE 40 -> 150, and gating every measurement on
its own null test. Drift is now <= 0.035 everywhere and no condition trips the gate.

Incidentally this exposed a real property: settling is ASYMMETRIC and much slower than t90 suggests.
A small step from equilibrium settles in ~5 steps, but a large excursion (0 -> -1.44) is still moving
after 40.

## Results (NREP=96, SETTLE=150, du=0.30, nominal v=22 roll=0 a=0 c0=0)

    sweep                  DC gain            dead time   note
    speed 5 -> 34          1.374 -> 1.827     4 -> 2      monotone; confirms G(v) ~ 0.0093v + 1.34
    long. accel -2 -> +2   1.379 -> 1.541     3           mild (~12%), ignorable
    roll -1.5 -> +1.5      1.400 -> 2.889     2-3         CONFOUNDED: u0 must counteract roll
    operating lataccel     1.415 -> 7.348     2-5         large -- see caveat
    linearity du +-0.6     1.37 - 1.69        2-3         no trend with |du| or sign

## Finding: gain depends on OPERATING POINT, not just speed

Sorting the roll and c0 sweeps by the operating steer they imply:

    u0 = -1.356 -> 2.418          u0 = +0.678 -> 1.936
    u0 = -1.017 -> 1.683          u0 = +1.017 -> 2.889
    u0 = -0.678 -> 1.637          u0 = +1.356 -> 7.348
    u0 =  0.000 -> 1.415

Gain rises with |steer| and much more steeply on the positive side. The `G(v)` schedule used by
`ff_pi`, `ff_pi_boot`, the DMC teacher and the `PMG` gain prior models NONE of this.

**Caveat: the 7.348 is probably not a vehicle property.** At c0=2.0 with du=+0.3 the plant is driven
toward ~4.2 lataccel, near the +-5 representable limit and in a region the training data barely
covers, so a learned plant extrapolating off-distribution is more likely than a 5x real gain change.
The sign asymmetry points the same way -- a physical steering gain should not care much about sign.

## Finding: linearity holds LOCALLY

Across du from -0.60 to +0.60 the DC gain is 1.37-1.69 with no trend in |du| or sign (scatter ~9%,
larger for small steps where signal/noise is worse). So superposition is sound for small deviations
about an operating point, which is what every linear design here assumes. What is NOT sound is
treating the operating-point gain as G(v) alone.

## Actionable

Schedule the gain on (v, lataccel) rather than v alone, and test it on `ff_pi_boot`, where G enters
explicitly in both the feedforward and the bootstrap anchor. Restrict the schedule to the |lataccel|
range the data actually covers, since the extreme-c0 measurements are suspect.

---

# Testing the asymmetry: real property, but correcting it as a constant fails

`controllers/ff_pi_asym.py` -- ff_pi_boot with a direction-dependent gain:

    G_eff = G(v) * (1 + asym)   when (desired - roll) > 0
    G_eff = G(v) * (1 - asym)   otherwise

applied to BOTH the feedforward and the bootstrap anchor, since both invert the same plant.
`asym=0` reproduces ff_pi_boot bit-for-bit (verified on 3 segments).

The measured ratio 1.614/1.390 = 1.16 implies asym ~ 0.074, predicted BEFORE any sweep.

## The sweep peaks exactly where the measurement predicted

Held-out ALL[500:1500], n=1000, vs asym=0 (52.200):

    asym=-0.150   +1.423 [-0.24,+2.92]  median +0.373  376/1000
    asym=-0.074   +0.116 [-1.23,+1.54]  median +0.141  437/1000
    asym=+0.037   -0.546 [-1.81,+0.42]  median -0.030  530/1000
    asym=+0.074   -1.143 [-3.27,+0.32]  median -0.084  541/1000   <- best, = predicted value
    asym=+0.111   -0.062 [-2.19,+1.47]  median -0.100  529/1000
    asym=+0.150   +0.957 [-1.42,+2.99]  median -0.031  511/1000

Wrong sign is worse, right sign better, with a peak at the independently predicted value and
degradation either side. That is a physical story rather than curve fitting -- but the CI already
included zero, so it was taken to confirmation rather than promoted.

## Confirmation: fails on the mean, on two independent splits

    headline ALL[:5000]        asym=+0.074 51.764 median 45.68 p90 78.16 | asym=0 51.222 45.66 76.75
                               mean +0.542 [-0.25,+1.31]  median -0.0687  better 2682/5000

    pristine ALL[5000:6000]    asym=+0.074 53.531 median 45.08 p90 80.52 | asym=0 52.930 45.23 75.91
                               mean +0.602 [-0.83,+2.16]  median -0.0452  better  521/1000

**Note the signature, which is the REVERSE of the usual trap in this project.** Every other
false positive here had mean-improves / median-worse. This has MEDIAN improves and SIGN TEST
favourable (2682/5000 = 53.6%, 5.1 sigma) while the MEAN gets worse and p90 degrades 6%. The
correction genuinely helps the typical segment and damages the tail.

## Why: the asymmetry is not constant

From the magnitude sweep, the asymmetry itself depends on operating point:

    c0 = 0.0   du=-0.6 -> 1.390   du=+0.6 -> 1.614     16%
    c0 = 1.0   du=-0.6 -> 1.578   du=+0.6 -> 2.232     41%

A fixed +7.4% correction is right for the common regime -- median |target lataccel| is 0.073, p90 is
0.720 -- and substantially wrong in the high-lataccel segments that dominate the tail. Hence better
median, worse p90, worse mean.

Scheduling `asym` on lataccel is the obvious refinement and is NOT recommended: the high-|lataccel|
measurements are the untrustworthy ones (only 0.8% of data has |lataccel| > 2, and the plant is a
learned model extrapolating there). Fitting that schedule would be fitting model extrapolation.

## The open diagnostic

Why does correcting a real, 20-sigma, 16% gain error buy nothing on the mean? Most likely the
INTEGRATOR was already absorbing it -- steady-state model error is exactly what integral action
exists to remove. That is testable: apply the same correction to the PID stack, whose integral
action is weaker, and the effect should be much larger. It distinguishes "the asymmetry does not
matter" from "the feedback already handled it", and those imply different things for everything
else here that inverts G.
