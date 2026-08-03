"""pid_boot + conditional anti-windup, so the aggressive "easy" integral gain becomes safe to use.

Diagnosis. Raising pid_boot's integral gain from i=0.100 to i=0.140 wins on 840 of 1000 pristine
segments and improves the median from 55.69 to 49.76 -- but the MEAN degrades 69.455 -> 93.090. The
damage is not spread out:

    worst  10 segments contribute +21.14 of the +23.63 mean delta   (89.5%)
    worst  25 segments contribute +27.69                            (117%, the rest being net negative)

and those segments have one unmistakable signature:

    group          n     mean d      J*    sat steps   max|integ|
    worst 25      25   +1107.46   35.85       29.0        15.60
    worst 26-160 135     +12.08    2.90        0.0         4.41
    rest         840      -6.77    2.76        0.0         3.44

**Rate-clamp saturation.** The failing segments spend 29 steps with the plant's MAX_ACC_DELTA clamp
binding; every other group spends zero. On those same 25 segments the nominal gain saturates for only
8 steps, so the higher integral gain is what drives the plant into the clamp. The runaway is textbook
windup: the clamp binds, the integrator keeps accumulating against a limit it cannot move, the
command overshoots, and the overshoot re-saturates. pid_boot's `i_clip` defaults to 1e9, so nothing
bounds it.

RESULT (pristine ALL[5000:6000], n=1000, vs nominal pid_boot):

    easy gains, no anti-windup    mean +23.634   median -5.93   better 840/1000
    easy gains + anti-windup      mean  +3.431   median -4.54   better 846/1000   <- 86% of tail damage removed
    nominal gains + anti-windup   mean  -0.580   median  0.00   better  18/1000

So the mechanism is confirmed, and it costs nothing at nominal gains (it almost never fires there).
A hold/bleed sweep on the separate tuning split ALL[3000:3800] put the optimum at the defaults --
hold=3 (the dead time), bleed=0 -- with every bleed>0 strictly worse.

Not promoted: the mean is still not below nominal. See `pid_fawu` for the scheduled+reactive stack.
"""
from ._awu import AWUMixin
from .pid_boot import Controller as _Boot


class Controller(AWUMixin, _Boot):
    pass
