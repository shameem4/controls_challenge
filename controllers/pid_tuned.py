"""Parameterised copy of the stock PID, for CMA-ES tuning.

controllers/pid.py is comma's reference baseline and is quoted throughout this repo
(110.76 on the full 5000), so it is left untouched. This is an exact functional copy whose
gains are constructor arguments; the defaults reproduce the stock controller bit-for-bit.
"""
from . import BaseController


class Controller(BaseController):
    def __init__(self, p=0.195, i=0.100, d=-0.053):
        self.p = p
        self.i = i
        self.d = d
        self.error_integral = 0
        self.prev_error = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff
