from dataclasses import dataclass, field
from typing import Any, List, Optional
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
class AdaptiveGains:
  """Container for adaptive PID gains"""
  p: float = 0.3
  i: float = 0.07
  d: float = -0.1
  feedforward_gain: float = 0.8


@dataclass
class Controller(BaseController):
  """PID controller with neural network-based gain adaptation."""

  # Base PID gains (can be overridden by neural network)
  p: float = 0.3
  i: float = 0.07
  d: float = -0.1

  # Feedforward parameters
  steer_factor: float = 13.0
  steer_sat_v: float = 20.0
  steer_command_sat: float = 2.0
  feedforward_gain: float = 0.8

  # Future blending
  future_weights: List[float] = field(default_factory=lambda: [5, 6, 7, 8])

  # PID scaling
  pid_target_scale_threshold: float = 1.0
  pid_target_scale_rate: float = 0.23
  longitudinal_gain_scale: float = 10.0

  # Velocity parameters
  minimum_velocity: float = 1.0

  # Neural network parameters
  use_adaptive_gains: bool = True
  gain_adapter: Optional[Any] = None
  history_length: int = 10

  # Internal state
  integrator: float = field(default=0.0, init=False)
  prev_error: float = field(default=0.0, init=False)
  control_steps: int = field(default=0, init=False)

  # History buffers for neural network
  error_history: List[float] = field(default_factory=list, init=False)
  lataccel_history: List[float] = field(default_factory=list, init=False)
  velocity_history: List[float] = field(default_factory=list, init=False)
  action_history: List[float] = field(default_factory=list, init=False)

  # Adaptive gains (updated by neural network)
  current_gains: AdaptiveGains = field(default_factory=AdaptiveGains, init=False)

  def reset(self) -> None:
    self.integrator = 0.0
    self.prev_error = 0.0
    self.control_steps = 0
    self.error_history.clear()
    self.lataccel_history.clear()
    self.velocity_history.clear()
    self.action_history.clear()
    self.current_gains = AdaptiveGains(p=self.p, i=self.i, d=self.d, feedforward_gain=self.feedforward_gain)

  def _update_history(self, error: float, current_lataccel: float, v_ego: float, action: float) -> None:
    """Maintain rolling history for neural network input."""
    self.error_history.append(error)
    self.lataccel_history.append(current_lataccel)
    self.velocity_history.append(v_ego)
    self.action_history.append(action)

    # Keep only recent history
    if len(self.error_history) > self.history_length:
      self.error_history.pop(0)
      self.lataccel_history.pop(0)
      self.velocity_history.pop(0)
      self.action_history.pop(0)

  def _extract_features(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any) -> np.ndarray:
    """Extract features for neural network."""
    features = []

    # Current state
    error = target_lataccel - current_lataccel
    features.extend([
      error,
      current_lataccel,
      target_lataccel,
      state.v_ego,
      state.a_ego,
      state.roll_lataccel,
      self.integrator,
      self.prev_error,
    ])

    # History features (padded with zeros if not enough history)
    for i in range(self.history_length):
      if i < len(self.error_history):
        features.append(self.error_history[-(i+1)])
      else:
        features.append(0.0)

    for i in range(self.history_length):
      if i < len(self.lataccel_history):
        features.append(self.lataccel_history[-(i+1)])
      else:
        features.append(0.0)

    for i in range(self.history_length):
      if i < len(self.velocity_history):
        features.append(self.velocity_history[-(i+1)])
      else:
        features.append(0.0)

    # Future plan features
    if future_plan and hasattr(future_plan, 'lataccel'):
      future_lataccels = future_plan.lataccel[:4]
      features.extend(future_lataccels + [0.0] * (4 - len(future_lataccels)))

      future_v_egos = future_plan.v_ego[:4] if hasattr(future_plan, 'v_ego') else [0.0] * 4
      features.extend(future_v_egos[:4] + [0.0] * (4 - len(future_v_egos[:4])))
    else:
      features.extend([0.0] * 8)

    # Road characteristics (derived from future plan)
    if future_plan and hasattr(future_plan, 'lataccel') and len(future_plan.lataccel) > 0:
      lataccel_variance = float(np.var(future_plan.lataccel[:4])) if len(future_plan.lataccel) >= 4 else 0.0
      lataccel_max = float(np.max(np.abs(future_plan.lataccel[:4]))) if len(future_plan.lataccel) >= 4 else 0.0
      features.extend([lataccel_variance, lataccel_max])
    else:
      features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)

  def _adapt_gains(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any) -> AdaptiveGains:
    """Use neural network to adapt gains based on current conditions."""
    if not self.use_adaptive_gains or self.gain_adapter is None:
      return AdaptiveGains(p=self.p, i=self.i, d=self.d, feedforward_gain=self.feedforward_gain)

    features = self._extract_features(target_lataccel, current_lataccel, state, future_plan)
    gains = self.gain_adapter.predict_gains(features)
    return gains

  def _blend_future_targets(self, current_target: float, future_plan: Any) -> float:
    future_targets = future_plan.lataccel if future_plan else []
    blended = _weighted_average([current_target] + future_targets, self.future_weights)
    return blended if blended else current_target

  def _pid(self, target_lataccel: float, current_lataccel: float, longitudinal_accel: float, gains: AdaptiveGains) -> float:
    error = target_lataccel - current_lataccel
    self.integrator += error
    error_diff = error - self.prev_error
    self.prev_error = error

    pid_factor = min(1.0, 1.0 - max(0.0, abs(target_lataccel) - self.pid_target_scale_threshold) * self.pid_target_scale_rate)
    proportional_gain = gains.p - abs(longitudinal_accel) / self.longitudinal_gain_scale
    return (proportional_gain * error + gains.i * self.integrator + gains.d * error_diff) * pid_factor

  def _feedforward(self, target_lataccel: float, state: Any, gains: AdaptiveGains) -> float:
    steer_accel_target = target_lataccel - state.roll_lataccel
    denom = max(self.steer_sat_v, max(self.minimum_velocity, state.v_ego))
    steer_command = (steer_accel_target * self.steer_factor) / denom
    steer_command = 2 * self.steer_command_sat / (1 + np.exp(-steer_command)) - self.steer_command_sat
    return gains.feedforward_gain * steer_command

  def compute_action(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any, control_state: ControlState) -> float:
    if future_plan and len(future_plan.lataccel) >= 3:
      target_lataccel = self._blend_future_targets(target_lataccel, future_plan)

    # Adapt gains based on current conditions
    self.current_gains = self._adapt_gains(target_lataccel, current_lataccel, state, future_plan)

    pid_term = self._pid(target_lataccel, current_lataccel, state.a_ego, self.current_gains)
    feedforward_term = self._feedforward(target_lataccel, state, self.current_gains)

    action = pid_term + feedforward_term

    # Update history for next iteration
    error = target_lataccel - current_lataccel
    self._update_history(error, current_lataccel, state.v_ego, action)

    self.control_steps += 1

    return action

  def get_diagnostics(self) -> dict:
    """Return current adaptive gains for analysis."""
    return {
      'adaptive_p': self.current_gains.p,
      'adaptive_i': self.current_gains.i,
      'adaptive_d': self.current_gains.d,
      'adaptive_ff_gain': self.current_gains.feedforward_gain,
    }
