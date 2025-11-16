from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from . import BaseController, ControlState


def _weighted_average(samples: List[float], weights: List[float]) -> float:
  valid_count = min(len(samples), len(weights))
  if valid_count == 0:
    return samples[0] if samples else 0.0
  samples = samples[:valid_count]
  weights = weights[:valid_count]
  return float(np.average(samples, weights=weights))


@dataclass
class Controller(BaseController):
  """Advanced PID controller with gain scheduling and adaptive feedforward."""

  p: float = 0.35
  i: float = 0.08
  d: float = -0.12
  future_weights: List[float] = field(default_factory=lambda: [6, 7, 8, 9])
  pid_target_scale_threshold: float = 1.0
  pid_target_scale_rate: float = 0.2
  longitudinal_gain_scale: float = 12.0
  minimum_velocity: float = 1.0
  steer_command_sat: float = 2.0
  base_feedforward_gain: float = 0.9
  feedforward_leak: float = 0.01
  speed_gain_schedule: Dict[float, float] = field(default_factory=lambda: {
    0: 12.0,
    10: 13.0,
    20: 14.0,
    30: 15.0,
    40: 16.5
  })
  adaptive_learning_rate: float = 0.02
  integrator: float = field(default=0.0, init=False)
  prev_error: float = field(default=0.0, init=False)
  last_feedforward_err: float = field(default=0.0, init=False)

  def reset(self) -> None:
    self.integrator = 0.0
    self.prev_error = 0.0
    self.last_feedforward_err = 0.0

  def _blend_future_targets(self, current_target: float, future_plan: Any) -> float:
    future_targets = future_plan.lataccel if future_plan else []
    blended = _weighted_average([current_target] + future_targets, self.future_weights)
    return blended if blended else current_target

  def _pid(self, target_lataccel: float, current_lataccel: float, longitudinal_accel: float, control_state: ControlState) -> float:
    error = target_lataccel - current_lataccel
    self.integrator += error
    self.integrator = float(np.clip(self.integrator, -8.0, 8.0))
    error_diff = error - self.prev_error
    self.prev_error = error

    pid_factor = min(1.0, 1.0 - max(0.0, abs(target_lataccel) - self.pid_target_scale_threshold) * self.pid_target_scale_rate)
    proportional_gain = (self.p - abs(longitudinal_accel) / self.longitudinal_gain_scale)
    proportional_gain = max(0.05, proportional_gain)
    return (proportional_gain * error + self.i * self.integrator + self.d * error_diff) * pid_factor

  def _lookup_steer_factor(self, v_ego: float) -> float:
    speeds = sorted(self.speed_gain_schedule.keys())
    if not speeds:
      return 12.0
    if v_ego <= speeds[0]:
      return self.speed_gain_schedule[speeds[0]]
    if v_ego >= speeds[-1]:
      return self.speed_gain_schedule[speeds[-1]]
    for low, high in zip(speeds[:-1], speeds[1:]):
      if low <= v_ego <= high:
        low_gain = self.speed_gain_schedule[low]
        high_gain = self.speed_gain_schedule[high]
        ratio = (v_ego - low) / (high - low)
        return low_gain + (high_gain - low_gain) * ratio
    return self.speed_gain_schedule[speeds[0]]

  def _feedforward(self, target_lataccel: float, state: Any) -> float:
    steer_factor = self._lookup_steer_factor(state.v_ego)
    steer_accel_target = target_lataccel - state.roll_lataccel
    denom = max(self.minimum_velocity, state.v_ego)
    steer_command = (steer_accel_target * steer_factor) / denom
    steer_command = 2 * self.steer_command_sat / (1 + np.exp(-steer_command)) - self.steer_command_sat
    return self.base_feedforward_gain * steer_command

  def _update_adaptive_terms(self, control_state: ControlState, target_lataccel: float, current_lataccel: float) -> None:
    prediction = control_state.history.latest('predicted_lataccel')
    if prediction is None:
      return
    actual_error = target_lataccel - current_lataccel
    prediction_error = target_lataccel - prediction
    delta = (prediction_error - self.last_feedforward_err) * self.adaptive_learning_rate
    self.last_feedforward_err = prediction_error
    self.base_feedforward_gain = float(np.clip((self.base_feedforward_gain * (1 - self.feedforward_leak)) + delta + 0.001 * actual_error, 0.5, 1.3))

  def compute_action(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any, control_state: ControlState) -> float:
    if future_plan and len(future_plan.lataccel) >= 3:
      target_lataccel = self._blend_future_targets(target_lataccel, future_plan)

    self._update_adaptive_terms(control_state, target_lataccel, current_lataccel)

    pid_term = self._pid(target_lataccel, current_lataccel, state.a_ego, control_state)
    feedforward_term = self._feedforward(target_lataccel, state)
    return pid_term + feedforward_term
