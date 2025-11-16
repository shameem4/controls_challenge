from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from . import BaseController, ControlState


@dataclass
class Controller(BaseController):
  """Adaptive PID controller that retunes gains during a rollout."""
  p: float = 0.2
  i: float = 0.12
  d: float = 0.01
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
  integral_limit: float = 1.5
  integral_leak: float = 0.05
  integral_flip_decay: float = 0.75
  max_derivative_ratio: float = 0.85
  derivative_floor: float = 0.02
  error_integral: float = field(default=0.0, init=False)
  prev_error: float = field(default=0.0, init=False)
  error_ma: float = field(default=0.0, init=False)
  jerk_ma: float = field(default=0.0, init=False)
  lat_cost_ma: float = field(default=0.0, init=False)
  _error_ma_initialized: bool = field(default=False, init=False)
  _jerk_ma_initialized: bool = field(default=False, init=False)
  _lat_cost_ma_initialized: bool = field(default=False, init=False)
  _last_tuned_step: int = field(default=0, init=False)
  _baseline_gains: Tuple[float, float, float] = field(init=False, repr=False)

  def __post_init__(self) -> None:
    self._baseline_gains = (self.p, self.i, self.d)

  def reset(self) -> None:
    """Clear any accumulated state so the controller can be reused."""
    self.p, self.i, self.d = self._baseline_gains
    self.error_integral = 0.0
    self.prev_error = 0.0
    self.error_ma = 0.0
    self.jerk_ma = 0.0
    self.lat_cost_ma = 0.0
    self._error_ma_initialized = False
    self._jerk_ma_initialized = False
    self._lat_cost_ma_initialized = False
    self._last_tuned_step = 0

  def _limit_integral(self) -> None:
    limit = self.integral_limit
    self.error_integral = max(-limit, min(limit, self.error_integral))

  def _compute_error_terms(self, target_lataccel: float, current_lataccel: float) -> Tuple[float, float]:
    """Return the proportional error and its first derivative."""
    error = target_lataccel - current_lataccel
    if self.error_integral != 0.0 and (self.error_integral > 0 > error or self.error_integral < 0 < error):
      self.error_integral *= (1 - self.integral_flip_decay)
    self.error_integral *= (1 - self.integral_leak)
    self.error_integral += error
    self._limit_integral()
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

  def _apply_adjustment(self, attr: str, delta: float, bounds: Tuple[float, float], max_step: Optional[float] = None) -> None:
    if delta == 0.0:
      return
    if max_step is not None and max_step > 0:
      delta = max(-max_step, min(max_step, delta))
    lower, upper = bounds
    new_value = getattr(self, attr) + delta
    setattr(self, attr, max(lower, min(upper, new_value)))

  def _limit_derivative_term(self, d_term: float, p_term: float, i_term: float) -> float:
    denom = abs(p_term) + abs(i_term) + self.derivative_floor
    max_term = self.max_derivative_ratio * denom
    return max(-max_term, min(max_term, d_term))

  def _reset_on_spike(self, lat_cost: Optional[float], jerk_cost: Optional[float], step_idx: int) -> bool:
    lat_trigger = lat_cost is not None and lat_cost > (self.error_target * 600)
    jerk_trigger = jerk_cost is not None and jerk_cost > (self.jerk_target * 6)
    if not (lat_trigger or jerk_trigger):
      return False
    self.p, self.i, self.d = self._baseline_gains
    self.error_integral = 0.0
    self.prev_error = 0.0
    self._limit_integral()
    self._last_tuned_step = step_idx
    return True

  def _recover_if_unstable(self) -> None:
    error_overflow = self.error_ma > (self.error_target * 12)
    jerk_overflow = self.jerk_ma > (self.jerk_target * 8)
    lat_cost_overflow = self.lat_cost_ma > (self.error_target * 800)
    if (error_overflow and jerk_overflow) or lat_cost_overflow:
      self.p, self.i, self.d = self._baseline_gains
      self.error_integral *= 0.5
      self._limit_integral()
      self._last_tuned_step = 0

  def compute_action(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any, control_state: ControlState) -> float:
    """
    Produce a steering command given the current lateral acceleration error.

    The future plan and detailed vehicle state are provided to enable more
    advanced controllers, but this baseline only needs the error terms.
    """
    error, error_diff = self._compute_error_terms(target_lataccel, current_lataccel)
    control_state.record('lat_error', error)
    control_state.record('error_diff', error_diff)
    p_term = self.p * error
    i_term = self.i * self.error_integral
    d_term = self.d * error_diff
    d_term = self._limit_derivative_term(d_term, p_term, i_term)
    control_state.record('pid_p_term', p_term)
    control_state.record('pid_i_term', i_term)
    control_state.record('pid_d_term', d_term)
    return p_term + i_term + d_term

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
    if self._reset_on_spike(lat_cost, jerk_cost, control_state.step_idx):
      return

    self._update_average('error_ma', '_error_ma_initialized', mean_abs_error)
    if lat_cost is not None:
      self._update_average('lat_cost_ma', '_lat_cost_ma_initialized', lat_cost)
      if lat_cost > (self.error_target * 400):
        self.error_integral *= 0.7
        self._limit_integral()
    if jerk_cost is not None:
      self._update_average('jerk_ma', '_jerk_ma_initialized', jerk_cost)

    stability_pressure = 0.0
    if jerk_cost is not None and self.jerk_target > 0.0:
      stability_pressure = max(0.0, (self.jerk_ma - self.jerk_target) / self.jerk_target)

    if self.error_target > 0.0:
      error_pressure = (self.error_ma - self.error_target) / self.error_target
      max_p_step = self.p_learning_rate * 4
      self._apply_adjustment('p', self.p_learning_rate * error_pressure, self.p_bounds, max_step=max_p_step)

    if abs(mean_error) > self.bias_target:
      bias_sign = 1.0 if mean_error > 0 else -1.0
      self._apply_adjustment('i', self.i_learning_rate * bias_sign, self.i_bounds, max_step=self.i_learning_rate * 3)
    else:
      decay = -self.i_learning_rate * 0.2 if self.i > self.i_bounds[0] else 0.0
      self._apply_adjustment('i', decay, self.i_bounds, max_step=self.i_learning_rate * 3)

    if jerk_cost is not None and self.jerk_target > 0.0:
      jerk_pressure = (self.jerk_ma - self.jerk_target) / self.jerk_target
      # When jerk is high we want a more negative derivative gain to dampen oscillations.
      self._apply_adjustment('d', -self.d_learning_rate * jerk_pressure, self.d_bounds, max_step=self.d_learning_rate * 4)

    if stability_pressure > 0.0:
      self._apply_adjustment('p', -self.p_learning_rate * 2.0 * min(2.0, stability_pressure), self.p_bounds, max_step=self.p_learning_rate * 3)
      self._apply_adjustment('i', -self.i_learning_rate * 0.5 * min(2.0, stability_pressure), self.i_bounds, max_step=self.i_learning_rate * 2)

    self._last_tuned_step = control_state.step_idx
    self._recover_if_unstable()

  def get_diagnostics(self) -> Dict[str, float]:
    return {
      'pid_p_gain': self.p,
      'pid_i_gain': self.i,
      'pid_d_gain': self.d,
      'pid_error_ma': self.error_ma,
      'pid_jerk_ma': self.jerk_ma,
      'pid_lat_cost_ma': self.lat_cost_ma
    }
