# A genuine fuzzy logic controller: the optimum of the family is the PI we already have

Branch `fuzzy`. Distinct from `pid_fuzzy` / `ff_pi_fuzzy`, which use a fuzzy membership to *schedule*
a PID's gains. Here the Takagi-Sugeno inference system IS the controller: a 25-rule base fires and
its weighted consequents produce the command.

## Design

Five triangular sets per input (NB NS ZE PS PB, centres -1..+1, width 0.5, verified partition of
unity). The classic skew-symmetric rule table, so the consequent depends on `(i-2)+(j-2)`. One
nonlinearity parameter:

    K_ij = OUT * sign(s) * |s|**GAMMA,    s = ((i-2)+(j-2))/4

`GAMMA=1` is linear, `<1` aggressive near zero error, `>1` gentle near zero. **This is the whole
point**: with linear consequents a TS system of this shape reduces algebraically to a PI, so GAMMA is
the only thing that can beat one.

The fuzzy block replaces **only** the `kp*e + ki*integ` term of `ff_pi_tau`, keeping the Tikhonov
reference, the tau-gated inverse-plant feedforward, the bootstrapped integrator and the rate-clamp
anti-windup. Those are worth ~4 points together, and `ff_pi_look`/`ff_pi_vlead` and two others were
null precisely because a second anticipation mechanism has nothing left to buy once a feedforward
exists — so a fuzzy block asked to also do the anticipating would be testing the wrong thing.
`fuzzy_off=True` reproduces `ff_pi_tau` bit-for-bit.

## Velocity form fails, for a textbook reason

The standard PI-like FLC is velocity form: inputs `(e, de)`, output `du`, accumulated. Tuned to
`ff_pi`'s exact equivalent gains it reproduces the MEDIAN (46.51 vs 46.09) but the MEAN blows up to
74.69.

Cause: in velocity form the proportional action also lives inside the accumulator. When the
anti-windup freeze or the clip binds, that proportional response is permanently lost, whereas a
positional PI recomputes `kp*e` every step. Classic velocity-form failure under clamping, landing on
exactly the hard segments that drive the mean.

A saturation hypothesis was tested first and refuted: scaling the input universe up at fixed
effective gain makes it slightly *worse* (57.14 -> 62.28) and the median is identical (51.05) at
every scale, so the typical segment never saturates.

## Positional form, and the actual result

Surface over `(error, integral)`; the integrator keeps `ff_pi`'s semantics exactly (clipped at
`i_clip`, frozen while the rate clamp binds). Gains matched analytically -- `e_scale/i_scale = ki/kp`
and `out = 2*kp*e_scale` give `kp=0.1424, ki=0.1353` to four decimals.

Tuning split `ALL[3000:3400]`, n=400:

| | mean | median |
|---|---|---|
| `ff_pi_tau` (identity gate) | **49.871** | 46.09 |
| positional fuzzy, PI-equivalent, GAMMA=1 | 50.774 | 46.78 |
| GAMMA=0.9 | 51.281 | 46.24 |
| GAMMA=1.1 | 52.386 | 47.98 |
| GAMMA=0.75 | 55.629 | 47.53 |
| GAMMA=1.3 | 57.601 | 50.69 |
| GAMMA=0.6 | 208.175 | 66.08 |
| GAMMA=1.6 | 71.175 | 58.87 |

**GAMMA=1 is the best member of the family.** The nonlinear control surface — the only thing a fuzzy
system offers over a PI here — is worse in both directions, sharply so below 0.75.

A separate 36-point grid over `(out, gamma, e_scale)` in velocity form found nothing better either:
best 53.465, and not one of the 36 beat the baseline.

Pure FLC with no feedforward, for reference: **169.6**, against `ff_pi_tau`'s 49.9. The fuzzy
inference contributes essentially nothing on its own; everything is in the feedforward.

## Conclusion

The fuzzy family's optimum is its linear member, which is the controller already shipped, and even
that carries a 0.9 penalty from the fuzzy machinery's own input clipping. Rejected.

What is genuinely untested is optimising all 25 consequents directly rather than shaping them with a
single GAMMA — that is a strictly larger family that contains PI. I would not expect it to pay: it is
a 25-dimensional search on an objective whose subset noise already produced a 16% phantom gain when
`ff_pi_tuned` was fitted on 60 segments, so the overfitting risk is high and the demonstrated
direction of the GAMMA sweep is against it.
