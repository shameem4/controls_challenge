# The cost, decomposed by frequency: 58% of the gap is one band, and it is a resonance

Prompted by reframing the trajectory as a note sequence with minimal discord. That maps onto
something exact rather than metaphorical: the benchmark cost **is** a frequency-weighted objective.

    lataccel term   5000 * mean((c - tau)^2)     flat in frequency
    jerk term       100  * mean((dc/DEL_T)^2)    a first difference -- a HIGH-PASS filter

By Parseval both are sums of per-frequency power, so cost attributes to bands with no approximation.
"Minimal discord" becomes a precise question: which bands carry our error, and which does the cost
punish.

## Result

128 pristine segments, `cnn_v4` against the seed-aware optimum (`steer_opt`'s best plan), same
segments and same noise draws. Parseval check: 50.28 from the spectrum against 50.30 direct.

| band (Hz) | tau power | cnn lat | cnn jerk | cnn total | opt total | gap | share |
|---|---|---|---|---|---|---|---|
| 0.00-0.10 | 0.2481 | 2.16 | 1.04 | 3.19 | 2.57 | 0.63 | 5.2% |
| 0.10-0.25 | 0.0291 | 4.75 | 2.07 | 6.82 | 4.65 | 2.17 | 18.0% |
| 0.25-0.50 | 0.0063 | 5.64 | 2.32 | 7.96 | 5.77 | 2.19 | 18.1% |
| **0.50-1.00** | 0.0022 | **9.74** | 4.38 | **14.12** | **7.09** | **7.03** | **58.3%** |
| 1.00-2.00 | 0.0008 | 4.48 | 5.23 | 9.71 | 7.60 | 2.11 | 17.5% |
| 2.00-3.50 | 0.0003 | 1.38 | 3.77 | 5.15 | 6.45 | -1.30 | -10.8% |
| 3.50-5.00 | 0.0001 | 0.48 | 2.84 | 3.32 | 4.09 | -0.77 | -6.4% |

**58% of the gap is in 0.5-1.0 Hz**, and the reference has essentially no content there -- 99.55% of
target power is below 1 Hz, and the 0.5-1.0 band holds 0.0022 against 0.2481 below 0.1 Hz. We are
producing error at a frequency nothing is asking us to follow.

That is not broadband noise. It is a **loop-shaping defect**, and it has a physical identity: the
plant is dead-time dominated with L ~ 0.25 s (`FINDINGS_SYSID.md`: K ~ 1.5, L ~ 0.25 s, T ~ 0.15 s,
L/T ~ 1.7), and the classic closed-loop resonance for dead time L sits at **f ~ 1/(4L) = 1 Hz**. The
loop is ringing at its own natural frequency, and the seed-aware optimum -- which can pre-compensate
-- does not.

Above 2 Hz `cnn_v4` is *better* than the optimum (negative gap): it is already smooth there, and
further smoothing is not the opportunity.

## Why this is worth acting on

Every previous decomposition -- by segment, by difficulty band, by headroom decile, by cost component
-- came back "diffuse, spread across ordinary driving, no concentration to target". This one is
concentrated: a single octave holds the majority of the recoverable cost, and it coincides with a
resonance the plant's own dead time predicts.

The natural intervention is loop shaping: attenuate the feedback path around 0.5-1.0 Hz (a notch, or
a lower-bandwidth feedback with the feedforward carrying more of the load). Unlike the gain-schedule
and difficulty-routing attempts, this targets a mechanism identified in the frequency domain rather
than a segment set identified by cost.

## Process note

The first version applied the analytic jerk weight `|1 - e^{-i w}|^2 / DEL_T^2` to the trajectory
spectrum. That assumes a CIRCULAR difference -- the FFT wraps `c[N-1]` back to `c[0]` and injects a
jump that is not in the cost -- and inflated the jerk term by 67% (36.20 against the true 21.67),
breaking the Parseval check at 64.88 vs 50.30. Taking the spectrum of the actual difference signal
fixes it exactly. The tracking half was correct throughout, which is what localised the bug.

---

## The 0.5-1 Hz peak is the Bode waterbed, not a fixable defect

The obvious intervention -- notch the feedback path in that band -- is a **null**, and the reason is
that it has the wrong sign.

`ff_pi_notch` adds a second-order RBJ notch on the feedback term (`depth=0` reproduces `ff_pi_tau`
bit-for-bit; the mirrored body was gated at 1e-6). Tuning split, 400 segments:

| config | mean | median |
|---|---|---|
| `ff_pi_tau` (depth=0) | **49.871** | **46.09** |
| f0=0.7 Q=2 depth=0.3 (best of 18) | 49.733 | 46.76 |
| f0=0.7 Q=2 depth=1.0 | 56.338 | 51.99 |
| f0=0.5 Q=1 depth=1.0 | 71.372 | 65.14 |

Best is -0.14 on the mean and *worse* on the median, and every deeper notch is monotonically worse.

**Why.** Error from disturbance is `S d` with `S = 1/(1+L)`. Reducing loop gain at a frequency RAISES
sensitivity there, so notching the feedback degrades disturbance rejection exactly where the error
lives. A notch only helps if the band contains self-generated ringing; here it contains amplified
plant noise.

Confirmed directly by sweeping the feedback gain and re-measuring the tracking-cost spectrum
(128 segments):

| gain | 0-0.1 Hz | 0.1-0.25 | 0.25-0.5 | 0.5-1.0 | 1.0-2.0 | 2.0-5.0 | total |
|---|---|---|---|---|---|---|---|
| 0.50 | **46.16** | 21.38 | 11.61 | **7.18** | 3.75 | 1.75 | 91.85 |
| 0.75 | 12.86 | 10.58 | 10.39 | 9.18 | 4.28 | 1.78 | 49.07 |
| **1.00** | 4.17 | 5.76 | 7.54 | 10.18 | 5.44 | 1.82 | **34.90** |
| 1.50 | **3.57** | 5.78 | 8.48 | **52.70** | 12.56 | 2.02 | 85.11 |
| 2.00 | 40.08 | 74.81 | 347.00 | 1052.80 | 58.80 | 9.07 | 1582.57 |

This is the waterbed exactly: low-frequency error falls monotonically with gain (46.16 -> 3.57) while
0.5-1.0 Hz rises monotonically (7.18 -> 10.18 -> 52.70 -> 1052.80). The peak **can** be removed --
gain 0.5 puts that band at 7.18, better than the seed-aware optimum's 7.09 -- but it costs 46.16 at
low frequency. Total cost is minimised at gain 1.0, the tuned value, so `ff_pi_tau` already sits at
the optimum of the trade.

## What this settles

The concentration is real but it is a **fundamental limit, not a defect**. Bode's integral says
sensitivity reduction at low frequency must be paid for above it, and dead time L ~ 0.25 s caps the
usable bandwidth near 1/(2L) ~ 2 Hz, so the payment lands at 0.5-1 Hz. That is where it appears.

It also explains cleanly why the seed-aware optimum wins there: it is **not causal feedback**. It
pre-compensates a noise draw it can see, and Bode's integral does not constrain feedforward from
future information. Consistent with `FINDINGS_SEED_ORACLE.md`, where the optimum's tracking (17.75)
sits below the causal lataccel floor of 19.50.

So the 58% of the gap concentrated in one octave is not addressable by any causal loop shaping. It is
the price of feedback under dead time, and both the classical controller and the CNN are already
paying it at the optimal rate.
