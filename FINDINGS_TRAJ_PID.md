# Trajectory-error PID: lataccel emergent from trajectory control

**Result: clean negative, and the reason is structural rather than a tuning failure.**

Idea: stop closing the loop on lateral acceleration. Reconstruct the trajectory the way the
visualisation does, drive a PID on trajectory error, and let lataccel be emergent.

`controllers/pid_traj.py`. Leaky double integration of the heading-error rate, exactly
`drive_template.html:laneOffsets`:

    psi <- psi * decay + (-(a_tgt - a_cur) / v) * dt      heading error, rad
    y   <- y   * decay + v * psi * dt                     lateral offset, m

plus a constant-error extrapolation to a lookahead point using `future_plan`, folded into a virtual
error with the units of lataccel so the parent's gains keep their meaning:

    e_traj = -(k_y * y_pred + k_psi * v * psi)
    e      = (1 - w) * e_lat + w * e_traj

`w = 0` reproduces stock `pid` bit-for-bit. **Identity gate: PASS** on 6 segments.

## The scaling had to be derived, not searched

The first sweep diverged (totals 4,513 to 64,766 against stock `pid`'s ~103). Not a tuning miss --
the grid was nowhere near unity loop gain. At steady state the leaky integrators settle at
`psi_ss = -e*tau/v` and `y_ss = -e*tau^2`, so

    e_traj / e_lat = k_y * tau^2 + k_psi * tau        (= 9*k_y + 3*k_psi at tau = 3)

The initial grid spanned DC gain 3.0 to 22.5. Every point ran 3-22x stock loop gain on top of two
extra integrators. Deriving the constraint first would have skipped the entire sweep.

## The trade is real, monotonic, and unaffordable

Tuning set `ALL[:60]`, `k_y=0.005 k_psi=0.16 i=0.1`:

| w | lat | jerk | total |
|---|---|---|---|
| 0.0 (stock pid) | 1.151 | 19.95 | **77.50** |
| 0.15 | 1.337 | 18.28 | 85.14 |
| 0.3 | 1.659 | 17.24 | 100.19 |
| 0.5 | 2.308 | 16.00 | 131.38 |
| 1.0 | 9.395 | 15.60 | 485.37 |

Held out, `ALL[5000:5200]`, n=200:

| config | lat | jerk | total | median |
|---|---|---|---|---|
| w=0 (stock pid) | 1.980 | 29.46 | **128.46** | 74.71 |
| w=0.15 | 2.199 | 24.84 | 134.78 | 83.28 |
| w=0.5 | 4.973 | 19.55 | 268.19 | 125.34 |
| w=1, tau=0.5 | 5.542 | 19.78 | 296.87 | 169.40 |
| w=1, tau=3 | 41.652 | 29.60 | 2112.18 | 389.28 |

**Jerk improves monotonically in w** -- 29.46 to 24.84 at w=0.15 (-16%), to 19.55 at w=0.5 (-34%).
The mechanism genuinely does produce smoother steering. But tracking degrades faster, and the cost
weights lataccel error 50x, so every step costs more than it buys.

Sweeping the leak constant at fixed DC gain is monotonic in the same direction (`tau` 0.5 -> 10
gives 191.87 -> 1956.65). As `tau -> 0` the virtual error degenerates back to plain lataccel error.
**Every gradient in the parameter space points back at the controller we started from.**

## Why it cannot work here

Two reasons, and the second is the deep one.

1. **Phase.** Position error is the double integral of lataccel error, so `w > 0` inserts two
   integrators into the error path. The plant is dead-time dominated (L ~ 0.25 s) and phase margin
   is already the binding constraint (`FINDINGS_SPECTRUM.md`). Integrators spend it freely.

2. **No new information.** `k_y` is nearly inert across every sweep -- heading error does all the
   work, and true position feedback contributes almost nothing. That is not a tuning artifact. The
   dataset contains no road, so "position error" is not measured; it is *reconstructed by
   integrating the same lataccel signal the controller already has*. It is an invertible transform
   of the existing input, carrying zero additional information, purchased at a real phase cost. An
   outer position loop pays for itself in a real vehicle because it closes on an independently
   sensed quantity (lane position from a camera). Here there is nothing to sense.

The visualisation motivated this and also explains it: `FINDINGS_VISUAL_VS_COST.md` showed the drawn
path is a heavily lowpassed view of the same signal. Controlling that view means controlling a
filtered copy of what we already control, with the filter's lag added.

## Worth keeping

The jerk reduction is real and reproduces held-out (-16% at w=0.15 for +11% tracking). If a variant
of this challenge ever weighted jerk more heavily than 2x per squared unit, this becomes viable at
some crossover. Not promoted; `pid_traj.py` and `traj_sweep.py` retained on the `traj` branch.
