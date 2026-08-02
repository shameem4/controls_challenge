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
