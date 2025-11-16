from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from . import BaseController, ControlState
from tinyphysics_core.config import DEL_T, LAT_ACCEL_COST_MULTIPLIER


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
  base_feedforward_gain: float = 1.0
  nominal_feedforward_gain: float = 1.0
  feedforward_leak: float = 0.1
  speed_gain_schedule: Dict[float, float] = field(default_factory=lambda: {
    0: 12.0,
    10: 13.0,
    20: 14.0,
    30: 15.0,
    40: 16.5
  })
  adaptive_learning_rate: float = 0.04
  prediction_threshold: float = 0.05
  integrator_freeze_threshold: float = 0.15
  jerk_threshold: float = 40.0
  integrator_clamp_limit: float = 4.0
  freeze_decay: float = 0.9
  jerk_decay_factor: float = 0.5
  saturation_decay: float = 0.85
  clamp_relief_error_threshold: float = 0.25
  clamp_relief_steps: int = 8
  clamp_relief_rate: float = 0.4
  mismatch_relief_rate: float = 0.15
  mismatch_feedforward_scale: float = 0.6
  integrator: float = field(default=0.0, init=False)
  prev_error: float = field(default=0.0, init=False)
  last_feedforward_err: float = field(default=0.0, init=False)
  last_feedforward_term: float = field(default=0.0, init=False)
  _feedforward_saturated: bool = field(default=False, init=False)
  last_lataccel: float = field(default=0.0, init=False)
  integrator_clamp_hits: int = field(default=0, init=False)
  integrator_freeze_steps: int = field(default=0, init=False)
  integrator_clamp_steps: int = field(default=0, init=False)
  pid_abs_sum: float = field(default=0.0, init=False)
  max_action_abs: float = field(default=0.0, init=False)
  lataccel_error_sum: float = field(default=0.0, init=False)
  jerk_error_sum: float = field(default=0.0, init=False)
  control_steps: int = field(default=0, init=False)
  _clamp_error_steps: int = field(default=0, init=False)
  clamp_relief_events: int = field(default=0, init=False)

  def reset(self) -> None:
    self.integrator = 0.0
    self.prev_error = 0.0
    self.last_feedforward_err = 0.0
    self.last_feedforward_term = 0.0
    self._feedforward_saturated = False
    self.last_lataccel = 0.0
    self.integrator_clamp_hits = 0
    self.integrator_freeze_steps = 0
    self.integrator_clamp_steps = 0
    self.pid_abs_sum = 0.0
    self.max_action_abs = 0.0
    self.lataccel_error_sum = 0.0
    self.jerk_error_sum = 0.0
    self.control_steps = 0
    self.base_feedforward_gain = self.nominal_feedforward_gain
    self._clamp_error_steps = 0
    self.clamp_relief_events = 0

  def _blend_future_targets(self, current_target: float, future_plan: Any) -> float:
    future_targets = future_plan.lataccel if future_plan else []
    blended = _weighted_average([current_target] + future_targets, self.future_weights)
    return blended if blended else current_target

  def _pid(self, target_lataccel: float, current_lataccel: float, longitudinal_accel: float, freeze_integrator: bool) -> float:
    error = target_lataccel - current_lataccel
    if freeze_integrator:
      self.integrator *= self.freeze_decay
      self.integrator_freeze_steps += 1
    else:
      self.integrator += error
    prev_integrator = self.integrator
    was_clamped = abs(prev_integrator) > self.integrator_clamp_limit
    clipped = float(np.clip(self.integrator, -self.integrator_clamp_limit, self.integrator_clamp_limit))
    if was_clamped:
      self.integrator_clamp_hits += 1
      self.integrator_clamp_steps += 1
      clipped *= self.saturation_decay
    self.integrator = clipped
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

  def _apply_recovery_controls(self, error: float, pid_term: float, feedforward_term: float) -> float:
    near_clamp = abs(self.integrator) >= self.integrator_clamp_limit * 0.9
    if near_clamp and abs(error) > self.clamp_relief_error_threshold:
      self._clamp_error_steps += 1
    else:
      self._clamp_error_steps = max(0, self._clamp_error_steps - 1)
    if self._clamp_error_steps >= self.clamp_relief_steps:
      self.base_feedforward_gain += (self.nominal_feedforward_gain - self.base_feedforward_gain) * self.clamp_relief_rate
      self.integrator *= self.freeze_decay
      self.clamp_relief_events += 1
      self._clamp_error_steps = 0
    action = pid_term + feedforward_term
    mismatch = (error * action) < 0 and abs(error) > self.clamp_relief_error_threshold
    if mismatch:
      feedforward_term *= self.mismatch_feedforward_scale
      self.base_feedforward_gain += (self.nominal_feedforward_gain - self.base_feedforward_gain) * self.mismatch_relief_rate
    self.base_feedforward_gain = float(np.clip(self.base_feedforward_gain, 0.7, 1.2))
    return feedforward_term

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

    jerk = abs(current_lataccel - self.last_lataccel) / DEL_T if self.control_steps > 0 else 0.0
    jerk_spike = jerk > self.jerk_threshold
    if jerk_spike:
      self.integrator *= self.jerk_decay_factor
    freeze_integrator = jerk_spike or (self._feedforward_saturated and abs(target_lataccel - current_lataccel) < self.integrator_freeze_threshold)

    pid_term = self._pid(target_lataccel, current_lataccel, state.a_ego, freeze_integrator)
    feedforward_term = self._feedforward(target_lataccel, state)
    error = target_lataccel - current_lataccel
    feedforward_term = self._apply_recovery_controls(error, pid_term, feedforward_term)
    action = pid_term + feedforward_term

    self.control_steps += 1
    self.pid_abs_sum += abs(pid_term)
    self.max_action_abs = max(self.max_action_abs, abs(action))
    error = target_lataccel - current_lataccel
    self.lataccel_error_sum += error**2
    if self.control_steps > 1:
      self.jerk_error_sum += jerk**2
    self.last_lataccel = current_lataccel
    return action

  def get_diagnostics(self) -> Dict[str, float]:
    avg_pid_term = self.pid_abs_sum / self.control_steps if self.control_steps else 0.0
    lataccel_cost = (self.lataccel_error_sum / max(1, self.control_steps)) * 100.0
    jerk_denominator = max(1, self.control_steps - 1)
    jerk_cost = (self.jerk_error_sum / jerk_denominator) * 100.0
    total_cost = lataccel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
    return {
      'base_feedforward_gain': self.base_feedforward_gain,
      'integrator': self.integrator,
      'last_feedforward_err': self.last_feedforward_err,
      'last_feedforward_term': self.last_feedforward_term,
      'feedforward_saturated': float(self._feedforward_saturated),
      'integrator_clamp_hits': float(self.integrator_clamp_hits),
      'integrator_freeze_steps': float(self.integrator_freeze_steps),
      'integrator_clamp_steps': float(self.integrator_clamp_steps),
      'avg_pid_term_abs': avg_pid_term,
      'max_action_abs': self.max_action_abs,
      'lataccel_cost_estimate': lataccel_cost,
      'jerk_cost_estimate': jerk_cost,
      'total_cost_estimate': total_cost,
      'clamp_relief_events': float(self.clamp_relief_events)
    }
