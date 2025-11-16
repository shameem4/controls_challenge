from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from . import BaseController, ControlState


@dataclass
class Controller(BaseController):
  """Adaptive PID controller that retunes gains during a rollout."""
  p: float = 0.2
  i: float = 0.12
  d: float = -0.06
  tuning_window: int = 25
  tuning_interval: int = 5
  error_target: float = 0.04
  bias_target: float = 0.01
  jerk_target: float = 120.0
  adaptation_alpha: float = 0.2
  p_learning_rate: float = 0.002
  i_learning_rate: float = 0.0004
  d_learning_rate: float = 0.001
  p_bounds: Tuple[float, float] = (0.05, 0.5)
  i_bounds: Tuple[float, float] = (0.0, 0.4)
  d_bounds: Tuple[float, float] = (-0.25, -0.005)
  error_integral: float = field(default=0.0, init=False)
  prev_error: float = field(default=0.0, init=False)
  error_ma: float = field(default=0.0, init=False)
  jerk_ma: float = field(default=0.0, init=False)
  lat_cost_ma: float = field(default=0.0, init=False)
  _error_ma_initialized: bool = field(default=False, init=False)
  _jerk_ma_initialized: bool = field(default=False, init=False)
  _lat_cost_ma_initialized: bool = field(default=False, init=False)
  _last_tuned_step: int = field(default=0, init=False)

  def reset(self) -> None:
    """Clear any accumulated state so the controller can be reused."""
    self.error_integral = 0.0
    self.prev_error = 0.0
    self.error_ma = 0.0
    self.jerk_ma = 0.0
    self.lat_cost_ma = 0.0
    self._error_ma_initialized = False
    self._jerk_ma_initialized = False
    self._lat_cost_ma_initialized = False
    self._last_tuned_step = 0

  def _compute_error_terms(self, target_lataccel: float, current_lataccel: float) -> Tuple[float, float]:
    """Return the proportional error and its first derivative."""
    error = target_lataccel - current_lataccel
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    return error, error_diff

  def _update_average(self, attr: str, flag_attr: str, new_value: float) -> float:
    initialized = getattr(self, flag_attr)
    if not initialized:
      setattr(self, flag_attr, True)
      setattr(self, attr, new_value)
      return new_value
    current = getattr(self, attr)
    updated = (1 - self.adaptation_alpha) * current + self.adaptation_alpha * new_value
    setattr(self, attr, updated)
    return updated

  def _apply_adjustment(self, attr: str, delta: float, bounds: Tuple[float, float]) -> None:
    if delta == 0.0:
      return
    lower, upper = bounds
    new_value = getattr(self, attr) + delta
    setattr(self, attr, max(lower, min(upper, new_value)))

  def compute_action(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any, control_state: ControlState) -> float:
    """
    Produce a steering command given the current lateral acceleration error.

    The future plan and detailed vehicle state are provided to enable more
    advanced controllers, but this baseline only needs the error terms.
    """
    error, error_diff = self._compute_error_terms(target_lataccel, current_lataccel)
    control_state.record('lat_error', error)
    control_state.record('error_diff', error_diff)
    return (self.p * error) + (self.i * self.error_integral) + (self.d * error_diff)

  def on_simulation_update(self, predicted_lataccel: float, control_state: ControlState, step_metrics: Optional[Dict[str, float]] = None) -> None:
    del predicted_lataccel
    if step_metrics is None or not control_state.is_control_active:
      return
    if control_state.step_idx - self._last_tuned_step < self.tuning_interval:
      return
    errors = control_state.history.recent('lat_error', self.tuning_window)
    if len(errors) < self.tuning_window:
      return

    mean_abs_error = sum(abs(err) for err in errors) / len(errors)
    mean_error = sum(errors) / len(errors)
    lat_cost = step_metrics.get('lataccel_cost')
    jerk_cost = step_metrics.get('jerk_cost')

    self._update_average('error_ma', '_error_ma_initialized', mean_abs_error)
    if lat_cost is not None:
      self._update_average('lat_cost_ma', '_lat_cost_ma_initialized', lat_cost)
    if jerk_cost is not None:
      self._update_average('jerk_ma', '_jerk_ma_initialized', jerk_cost)

    if self.error_target > 0.0:
      error_pressure = (self.error_ma - self.error_target) / self.error_target
      self._apply_adjustment('p', self.p_learning_rate * error_pressure, self.p_bounds)

    if abs(mean_error) > self.bias_target:
      bias_sign = 1.0 if mean_error > 0 else -1.0
      self._apply_adjustment('i', self.i_learning_rate * bias_sign, self.i_bounds)
    else:
      decay = -self.i_learning_rate * 0.2 if self.i > self.i_bounds[0] else 0.0
      self._apply_adjustment('i', decay, self.i_bounds)

    if jerk_cost is not None and self.jerk_target > 0.0:
      jerk_pressure = (self.jerk_ma - self.jerk_target) / self.jerk_target
      # When jerk is high we want a more negative derivative gain to dampen oscillations.
      self._apply_adjustment('d', -self.d_learning_rate * jerk_pressure, self.d_bounds)

    self._last_tuned_step = control_state.step_idx

  def get_diagnostics(self) -> Dict[str, float]:
    return {
      'pid_p_gain': self.p,
      'pid_i_gain': self.i,
      'pid_d_gain': self.d,
      'pid_error_ma': self.error_ma,
      'pid_jerk_ma': self.jerk_ma,
      'pid_lat_cost_ma': self.lat_cost_ma
    }
