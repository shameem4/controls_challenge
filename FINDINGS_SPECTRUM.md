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
