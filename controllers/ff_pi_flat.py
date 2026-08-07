"""ff_pi_tau with a FLAT feedforward gain, because the measured plant gain does not depend on speed.

`verify_plant.py` measures the plant's true steady-state DC gain by stepping the action from
equilibrium in `expected` mode (the conditional mean, so no sampling noise at all):

    v (m/s)      8      15      22      28      34
    measured   2.445   2.533   2.243   2.424   2.315      <- essentially FLAT
    G(v) used  1.422   1.421   1.475   1.563   1.692      <- rises with speed
    ratio      1.72    1.78    1.52    1.55    1.37

Two separate errors in the model we had been using.

**Magnitude.** `gain_fit.npy` is low by roughly 1.6x. That is what `ff_pi`'s tuned `gain_scale=1.79`
has been silently absorbing all along -- it was correcting a systematic identification error, not
detuning the loop as its name suggests. This also explains why `gain_scale` needed tuning at all.

**Slope.** The measured gain is flat across 8-34 m/s while `gain_fit.npy` rises, so the ratio falls
monotonically with speed. The speed dependence we modelled is not there. It is consistent with
`FINDINGS_SYSID.md`, which found the gain depends on the OPERATING POINT rather than on speed -- and
the operating-point dependence is already handled, by `ff_pi_tau`'s tau-gated detune.

So: replace `G(v)` with a constant, and let the tau gate keep doing the operating-point work. The
constant is applied multiplicatively through `gain_scale / gs_norm` so the tau modulation survives
untouched -- an earlier version of this test overrode `G()` to return a bare constant, which silently
disabled the tau detune and made flat gain look worse than it is.

`flat=None` restores `ff_pi_tau` exactly.
"""
from .ff_pi_tau import Controller as _FFTau


class Controller(_FFTau):
    def __init__(self, flat=2.5, **kw):
        kw.setdefault('gs_hard', 2.1)
        super().__init__(**kw)
        self.flat = None if flat is None else float(flat)

    def G(self, v):
        if self.flat is None:
            return super().G(v)
        # flat in v, but the tau gate still modulates it exactly as in ff_pi_tau
        return self.flat * (self.gain_scale / self.gs_norm)
