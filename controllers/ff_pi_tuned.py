"""ff_pi with CMA-ES-tuned gains (400-segment tuning set, held-out verified).

Tuned on ALL[2000:2400], validated on the disjoint clean split [500:620] where it scores
53.40 vs the hand-tuned defaults' 63.06 (-15.3%). An earlier attempt tuning on only 60
segments produced a 16% *apparent* gain that was almost entirely overfitting -- this metric's
subset noise is large enough that small tuning sets fit the sample, not the controller.

Notable shifts from the hand-tuned defaults: much stronger feedforward (gain_scale 1.3 -> 1.79),
and much heavier reference smoothing (lam 2.005 -> 4.74) than the analytically cost-optimal
value -- the analytic lambda assumes perfect tracking, so a real controller on a noisy plant
wants more.
"""
from .ff_pi import Controller as _FFPI

PARAMS = dict(kp=0.142366145934493, ki=0.13525350720368765, lead=2,
              gain_scale=1.7900104013982312, i_clip=3.1122763582436304,
              lam=4.736532585937806)


class Controller(_FFPI):
    def __init__(self, **kw):
        p = dict(PARAMS); p.update(kw)
        super().__init__(**p)
