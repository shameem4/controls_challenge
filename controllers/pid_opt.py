from dataclasses import dataclass, field
from typing import Any, List

import numpy as np

from . import BaseController, ControlState


def _weighted_average(samples: List[float], weights: List[float]) -> float:
  valid_count = min(len(samples), len(weights))
  if valid_count == 0:
    return 0.0
  samples = samples[:valid_count]
  weights = weights[:valid_count]
  return float(np.average(samples, weights=weights))


@dataclass
class Controller(BaseController):
  """PID controller with a simple feedforward term for higher-frequency response."""
  p: float = 0.3
  i: float = 0.07
  d: float = -0.1
  steer_factor: float = 13.0
  steer_sat_v: float = 20.0
  steer_command_sat: float = 2.0
  feedforward_gain: float = 0.8
  future_weights: List[float] = field(default_factory=lambda: [5, 6, 7, 8])
  pid_target_scale_threshold: float = 1.0
  pid_target_scale_rate: float = 0.23
  longitudinal_gain_scale: float = 10.0
  minimum_velocity: float = 1.0
  integrator: float = field(default=0.0, init=False)
  prev_error: float = field(default=0.0, init=False)
  control_steps: int = field(default=0, init=False)

  def reset(self) -> None:
    self.integrator = 0.0
    self.prev_error = 0.0
    self.control_steps = 0

  def _blend_future_targets(self, current_target: float, future_plan: Any) -> float:
    future_targets = future_plan.lataccel if future_plan else []
    blended = _weighted_average([current_target] + future_targets, self.future_weights)
    return blended if blended else current_target

  def _pid(self, target_lataccel: float, current_lataccel: float, longitudinal_accel: float) -> float:
    error = target_lataccel - current_lataccel
    self.integrator += error
    error_diff = error - self.prev_error
    self.prev_error = error

    pid_factor = min(1.0, 1.0 - max(0.0, abs(target_lataccel) - self.pid_target_scale_threshold) * self.pid_target_scale_rate)
    proportional_gain = self.p - abs(longitudinal_accel) / self.longitudinal_gain_scale
    return (proportional_gain * error + self.i * self.integrator + self.d * error_diff) * pid_factor

  def _feedforward(self, target_lataccel: float, state: Any) -> float:
    steer_accel_target = target_lataccel - state.roll_lataccel
    denom = max(self.steer_sat_v, max(self.minimum_velocity, state.v_ego))
    steer_command = (steer_accel_target * self.steer_factor) / denom
    steer_command = 2 * self.steer_command_sat / (1 + np.exp(-steer_command)) - self.steer_command_sat
    return self.feedforward_gain * steer_command

  def compute_action(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any, control_state: ControlState) -> float:

    if future_plan and len(future_plan.lataccel) >= 3:
      target_lataccel = self._blend_future_targets(target_lataccel, future_plan)

    pid_term = self._pid(target_lataccel, current_lataccel, state.a_ego)
    feedforward_term = self._feedforward(target_lataccel, state)
    return pid_term + feedforward_term
