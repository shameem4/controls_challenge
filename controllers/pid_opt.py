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
  nominal_feedforward_gain: float = 0.9
  feedforward_leak: float = 0.05
  speed_gain_schedule: Dict[float, float] = field(default_factory=lambda: {
    0: 12.0,
    10: 13.0,
    20: 14.0,
    30: 15.0,
    40: 16.5
  })
  adaptive_learning_rate: float = 0.01
  prediction_threshold: float = 0.05
  integrator_freeze_threshold: float = 0.15
  jerk_threshold: float = 5.0
  integrator: float = field(default=0.0, init=False)
  prev_error: float = field(default=0.0, init=False)
  last_feedforward_err: float = field(default=0.0, init=False)
  last_feedforward_term: float = field(default=0.0, init=False)
  _feedforward_saturated: bool = field(default=False, init=False)
  last_lataccel: float = field(default=0.0, init=False)
  integrator_clamp_hits: int = field(default=0, init=False)
  integrator_freeze_steps: int = field(default=0, init=False)

  def reset(self) -> None:
    self.integrator = 0.0
    self.prev_error = 0.0
    self.last_feedforward_err = 0.0
    self.last_feedforward_term = 0.0
    self._feedforward_saturated = False
    self.last_lataccel = 0.0
    self.integrator_clamp_hits = 0
    self.integrator_freeze_steps = 0

  def _blend_future_targets(self, current_target: float, future_plan: Any) -> float:
    future_targets = future_plan.lataccel if future_plan else []
    blended = _weighted_average([current_target] + future_targets, self.future_weights)
    return blended if blended else current_target

  def _pid(self, target_lataccel: float, current_lataccel: float, longitudinal_accel: float, freeze_integrator: bool) -> float:
    error = target_lataccel - current_lataccel
    if freeze_integrator:
      self.integrator *= 0.9
      self.integrator_freeze_steps += 1
    else:
      self.integrator += error
    prev_integrator = self.integrator
    self.integrator = float(np.clip(self.integrator, -4.0, 4.0))
    if self.integrator != prev_integrator:
      self.integrator_clamp_hits += 1
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
    self._feedforward_saturated = abs(steer_command) >= self.steer_command_sat * 0.98
    term = self.base_feedforward_gain * steer_command
    self.last_feedforward_term = term
    return term

  def _update_adaptive_terms(self, control_state: ControlState, target_lataccel: float, current_lataccel: float) -> None:
    prediction = control_state.history.latest('predicted_lataccel')
    if prediction is None:
      return
    actual_error = target_lataccel - current_lataccel
    prediction_error = target_lataccel - prediction
    if abs(prediction_error) > self.prediction_threshold:
      delta = (prediction_error - self.last_feedforward_err) * self.adaptive_learning_rate
      self.base_feedforward_gain += delta + 0.001 * actual_error
    else:
      self.base_feedforward_gain += (self.nominal_feedforward_gain - self.base_feedforward_gain) * self.feedforward_leak
    self.last_feedforward_err = prediction_error
    self.base_feedforward_gain = float(np.clip(self.base_feedforward_gain, 0.7, 1.2))

  def compute_action(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any, control_state: ControlState) -> float:
    if future_plan and len(future_plan.lataccel) >= 3:
      target_lataccel = self._blend_future_targets(target_lataccel, future_plan)

    self._update_adaptive_terms(control_state, target_lataccel, current_lataccel)

    jerk = abs(current_lataccel - self.last_lataccel)
    self.last_lataccel = current_lataccel
    freeze_integrator = jerk > self.jerk_threshold or (self._feedforward_saturated and abs(target_lataccel - current_lataccel) < self.integrator_freeze_threshold)

    pid_term = self._pid(target_lataccel, current_lataccel, state.a_ego, freeze_integrator)
    feedforward_term = self._feedforward(target_lataccel, state)
    return pid_term + feedforward_term

  def get_diagnostics(self) -> Dict[str, float]:
    return {
      'base_feedforward_gain': self.base_feedforward_gain,
      'integrator': self.integrator,
      'last_feedforward_err': self.last_feedforward_err,
      'last_feedforward_term': self.last_feedforward_term,
      'feedforward_saturated': float(self._feedforward_saturated),
      'integrator_clamp_hits': float(self.integrator_clamp_hits),
      'integrator_freeze_steps': float(self.integrator_freeze_steps)
    }
