from dataclasses import dataclass, field
from typing import Any, Tuple

from . import BaseController


@dataclass
class Controller(BaseController):
  """Basic PID controller used as the default baseline."""
  p: float = 0.195
  i: float = 0.100
  d: float = -0.053
  error_integral: float = field(default=0.0, init=False)
  prev_error: float = field(default=0.0, init=False)

  def reset(self) -> None:
    """Clear any accumulated state so the controller can be reused."""
    self.error_integral = 0.0
    self.prev_error = 0.0

  def _compute_error_terms(self, target_lataccel: float, current_lataccel: float) -> Tuple[float, float]:
    """Return the proportional error and its first derivative."""
    error = target_lataccel - current_lataccel
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    return error, error_diff

  def update(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any) -> float:
    """
    Produce a steering command given the current lateral acceleration error.

    The future plan and detailed vehicle state are provided to enable more
    advanced controllers, but this baseline only needs the error terms.
    """
    error, error_diff = self._compute_error_terms(target_lataccel, current_lataccel)
    return (self.p * error) + (self.i * self.error_integral) + (self.d * error_diff)
