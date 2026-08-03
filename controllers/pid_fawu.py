"""pid_fuzzy (difficulty-scheduled gains) + conditional anti-windup.

The two mechanisms address the same failure -- the aggressive "easy" integral gain driving the plant
into its rate clamp -- from opposite sides, so they are worth stacking:

  * `pid_fuzzy` is ANTICIPATORY and preview-based: it backs the gain off wherever J* says the upcoming
    trajectory is hard. That fires on hundreds of segments, most of which never actually saturate, so
    it gives up median performance broadly (median 53.31 vs the easy gains' 49.76).
  * anti-windup is REACTIVE: it fires only on the ~25 segments where the clamp is genuinely binding.
    Alone on the easy gains it recovers 86% of the mean damage (+23.6 -> +3.4) and keeps the full
    median win, but the residual tail is still positive.

Stacked, the scheduler can afford a much milder backoff because anti-windup catches whatever it
misses. Defaults are pid_fuzzy's.
"""
from ._awu import AWUMixin
from .pid_fuzzy import Controller as _Fuzzy


class Controller(AWUMixin, _Fuzzy):
    pass
