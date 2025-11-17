from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import BaseController, ControlState


def _weighted_average(samples: List[float], weights: List[float]) -> float:
  """Compute weighted average of samples."""
  valid_count = min(len(samples), len(weights))
  if valid_count == 0:
    return 0.0
  samples = samples[:valid_count]
  weights = weights[:valid_count]
  return float(np.average(samples, weights=weights))


@dataclass
class SegmentCharacteristics:
  """Estimated characteristics of the current driving segment."""
  avg_v_ego: float = 0.0
  std_v_ego: float = 0.0
  avg_abs_roll_lataccel: float = 0.0
  avg_abs_target_lataccel: float = 0.0
  avg_abs_target_lataccel_change: float = 0.0
  max_abs_target_lataccel: float = 0.0
  max_abs_target_change: float = 0.0  # NEW: Track maximum single target jump
  velocity_target_product: float = 0.0  # NEW: v_ego * |Δtarget| - difficulty metric
  target_variability: float = 0.0  # NEW: Standard deviation of targets
  num_large_target_changes: int = 0  # NEW: Count of changes > 0.05 m/s²
  num_samples: int = 0

  def update_online(self, v_ego: float, roll_lataccel: float, target_lataccel: float, prev_target_lataccel: Optional[float]) -> None:
    """Update running statistics with new observation."""
    n = self.num_samples

    # Update velocity statistics (Welford's online algorithm)
    delta = v_ego - self.avg_v_ego
    self.avg_v_ego += delta / (n + 1)
    if n > 0:
      # Running variance approximation
      self.std_v_ego = np.sqrt((n * self.std_v_ego**2 + delta * (v_ego - self.avg_v_ego)) / (n + 1))

    # Update roll lateral acceleration
    self.avg_abs_roll_lataccel = (self.avg_abs_roll_lataccel * n + abs(roll_lataccel)) / (n + 1)

    # Update target lateral acceleration
    abs_target = abs(target_lataccel)
    self.avg_abs_target_lataccel = (self.avg_abs_target_lataccel * n + abs_target) / (n + 1)
    self.max_abs_target_lataccel = max(self.max_abs_target_lataccel, abs_target)

    # Update target variability (running standard deviation)
    if n > 0:
      delta_target = target_lataccel - (self.avg_abs_target_lataccel if n == 1 else 0)  # Simplified for online calc
      self.target_variability = np.sqrt((n * self.target_variability**2 + abs_target**2) / (n + 1))

    # Update target change rate
    if prev_target_lataccel is not None:
      target_change = abs(target_lataccel - prev_target_lataccel)
      self.avg_abs_target_lataccel_change = (self.avg_abs_target_lataccel_change * n + target_change) / (n + 1)
      self.max_abs_target_change = max(self.max_abs_target_change, target_change)

      # NEW: Track combined difficulty metric
      combined_difficulty = v_ego * target_change
      self.velocity_target_product = (self.velocity_target_product * n + combined_difficulty) / (n + 1)

      # NEW: Count large target changes
      if target_change > 0.05:
        self.num_large_target_changes += 1

    self.num_samples += 1


@dataclass
class Controller(BaseController):
  """
  Adaptive PID controller with feedforward that estimates segment characteristics and adjusts gains.

  Key adaptive strategies based on correlation analysis:
  1. Velocity adaptation: Adjust gains based on speed (high correlation for pid_w_ff)
  2. Roll compensation: Scale control effort with roll-induced lateral acceleration
  3. Target aggressiveness: Dampen response when targets change rapidly
  4. Integral management: Limit windup based on segment dynamics
  5. Feedforward control: Use velocity-dependent feedforward for better tracking
  """

  # Base PID gains (tuned for use with feedforward)
  p_base: float = 0.24
  i_base: float = 0.08
  d_base: float = -0.09

  # Adaptive gain limits
  p_min: float = 0.18
  p_max: float = 0.30
  i_min: float = 0.05
  i_max: float = 0.12
  d_min: float = -0.12
  d_max: float = -0.05

  # Feedforward parameters (simplified from pid_w_ff)
  steer_factor: float = 13.0
  steer_sat_v: float = 20.0
  steer_command_sat: float = 2.0
  feedforward_gain_base: float = 0.75  # Slightly lower than pid_w_ff
  minimum_velocity: float = 1.0

  # Future plan blending (optional - use if available)
  future_weights: List[float] = field(default_factory=lambda: [5, 6, 7, 8])
  use_future_blending: bool = True

  # Adaptation parameters (conservative)
  velocity_scale_threshold: float = 25.0  # m/s
  velocity_scale_rate: float = 0.003  # Small adjustments
  high_roll_threshold: float = 0.5  # m/s^2
  aggressive_target_threshold: float = 0.015  # m/s^2 per step
  max_integral_base: float = 8.0  # Smaller base since feedforward helps
  integral_decay_rate: float = 0.98

  # NEW: High-difficulty scenario thresholds (from analysis)
  extreme_target_threshold: float = 2.0  # m/s² - extreme lateral demand (raised threshold)
  very_aggressive_threshold: float = 0.025  # m/s²/step - very rapid changes (raised)
  high_speed_threshold: float = 28.0  # m/s - high speed
  velocity_target_difficulty_threshold: float = 0.5  # Combined difficulty metric (raised)

  # NEW: High-demand scenario thresholds (where pid_w_ff excels)
  # Based on analysis: FF wins when avg_abs_target +38.9%, max_abs_target +79%
  high_demand_avg_threshold: float = 0.25  # m/s² avg target (FF wins at 0.297 vs 0.214)
  high_demand_max_threshold: float = 1.2  # m/s² max target (FF wins at 1.565 vs 0.874)

  # Estimation window for segment characteristics
  estimation_window: int = 35  # Build confidence before adapting

  # Internal state
  error_integral: float = field(default=0.0, init=False)
  prev_error: float = field(default=0.0, init=False)
  prev_target_lataccel: Optional[float] = field(default=None, init=False)
  segment_chars: SegmentCharacteristics = field(default_factory=SegmentCharacteristics, init=False)

  # Running buffers for recent history
  recent_v_ego: List[float] = field(default_factory=list, init=False)
  recent_targets: List[float] = field(default_factory=list, init=False)
  recent_errors: List[float] = field(default_factory=list, init=False)

  # Adaptive gains (computed)
  p_adaptive: float = field(default=0.24, init=False)
  i_adaptive: float = field(default=0.08, init=False)
  d_adaptive: float = field(default=-0.09, init=False)
  feedforward_gain_adaptive: float = field(default=0.75, init=False)

  def reset(self) -> None:
    """Clear any accumulated state so the controller can be reused."""
    self.error_integral = 0.0
    self.prev_error = 0.0
    self.prev_target_lataccel = None
    self.segment_chars = SegmentCharacteristics()
    self.recent_v_ego = []
    self.recent_targets = []
    self.recent_errors = []
    self.p_adaptive = self.p_base
    self.i_adaptive = self.i_base
    self.d_adaptive = self.d_base
    self.feedforward_gain_adaptive = self.feedforward_gain_base

  def _update_segment_characteristics(self, target_lataccel: float, state: Any) -> None:
    """Update running estimates of segment characteristics."""
    self.segment_chars.update_online(
      v_ego=state.v_ego,
      roll_lataccel=state.roll_lataccel,
      target_lataccel=target_lataccel,
      prev_target_lataccel=self.prev_target_lataccel
    )

    # Update recent history buffers (keep last 20 samples)
    self.recent_v_ego.append(state.v_ego)
    self.recent_targets.append(target_lataccel)
    if len(self.recent_v_ego) > 20:
      self.recent_v_ego.pop(0)
      self.recent_targets.pop(0)

    self.prev_target_lataccel = target_lataccel

  def _blend_future_targets(self, current_target: float, future_plan: Any) -> float:
    """Blend current target with future plan for anticipatory control."""
    if not self.use_future_blending or not future_plan:
      return current_target
    future_targets = future_plan.lataccel if future_plan else []
    if len(future_targets) < 3:
      return current_target
    blended = _weighted_average([current_target] + future_targets, self.future_weights)
    return blended if blended else current_target

  def _compute_adaptive_gains(self, state: Any) -> Tuple[float, float, float, float]:
    """
    Compute adaptive PID and feedforward gains based on segment characteristics.

    Strategy:
    - High velocity -> Rely more on feedforward, less on PID
    - High roll -> Increase P gain (more responsive)
    - Aggressive targets -> Lower I gain (less windup), higher D (smoother)
    - Low velocity variance -> Can increase I gain (more stable)
    - NEW: Extreme scenarios -> Emergency damping mode
    """
    chars = self.segment_chars

    # Start with base gains
    p_gain = self.p_base
    i_gain = self.i_base
    d_gain = self.d_base
    ff_gain = self.feedforward_gain_base

    # Only start adapting after we have enough samples
    if chars.num_samples < self.estimation_window:
      return p_gain, i_gain, d_gain, ff_gain

    # NEW: Strategy 0b - Detect extreme high-difficulty scenarios
    # Only trigger on truly extreme cases - be very selective
    # Priority: If extreme, skip high-demand mode later
    is_extreme_scenario = (
      (chars.max_abs_target_lataccel > self.extreme_target_threshold and
       chars.avg_abs_target_lataccel_change > self.aggressive_target_threshold) or
      chars.avg_abs_target_lataccel_change > self.very_aggressive_threshold
    )

    if is_extreme_scenario:
      # Extreme scenario mode: subtle adjustments only + modest FF boost
      i_gain *= 0.75  # Reduce integral gain to prevent windup
      d_gain *= 1.10  # Increase damping slightly
      ff_gain *= 1.12  # Modest FF boost (less aggressive than high-demand mode)

    # Strategy 1: Velocity-based adaptation - balance PID vs feedforward
    # At high speeds, feedforward is more effective; at low speeds, PID is better
    if chars.avg_v_ego > 28.0:  # High speed - rely more on feedforward
      ff_gain *= 1.1
      p_gain *= 0.95  # Reduce PID aggressiveness
      i_gain *= 0.9
    elif chars.avg_v_ego < 15.0:  # Low speed - rely more on PID
      ff_gain *= 0.85
      p_gain *= 1.05
      i_gain *= 1.05

    # Strategy 2: Roll compensation - feedforward handles this better
    # High roll situations - boost feedforward and P slightly
    if chars.avg_abs_roll_lataccel > self.high_roll_threshold:
      roll_factor = min(1.5, chars.avg_abs_roll_lataccel / self.high_roll_threshold)
      ff_gain *= (1.0 + 0.08 * (roll_factor - 1.0))  # Increase FF by up to 8%
      p_gain *= (1.0 + 0.04 * (roll_factor - 1.0))  # Increase P by up to 4%

    # Strategy 3: Target aggressiveness - key insight from analysis
    # This is where both controllers struggle - reduce I, increase D and FF
    if chars.avg_abs_target_lataccel_change > self.aggressive_target_threshold:
      aggression_factor = min(2.5, chars.avg_abs_target_lataccel_change / self.aggressive_target_threshold)
      i_gain *= max(0.7, 1.0 - 0.2 * (aggression_factor - 1.0))  # Reduce I by up to 30%
      d_gain *= (1.0 + 0.08 * (aggression_factor - 1.0))  # Increase D magnitude
      ff_gain *= (1.0 + 0.05 * (aggression_factor - 1.0))  # Slight FF boost for responsiveness

    # Strategy 4: Boost I slightly in very stable conditions
    if chars.std_v_ego < 2.0 and chars.avg_abs_target_lataccel_change < self.aggressive_target_threshold * 0.5:
      i_gain *= 1.08  # Small increase in stable conditions

    # NEW: Strategy 5 - Detect and handle extreme target variability
    # High target variability with many large changes indicates challenging segment
    if chars.target_variability > 0.4 and chars.num_large_target_changes > 20:
      i_gain *= 0.8  # Further reduce integral to prevent accumulation
      d_gain *= 1.15  # More damping for stability

    # NEW: Strategy 6 - HIGH-PRIORITY: High-demand scenarios (where pid_w_ff wins)
    # Based on analysis: FF excels when avg_abs_target > 0.25 AND max_abs_target > 1.2
    # This runs LAST to override other strategies in high-demand situations
    # UNLESS it's an extreme scenario (which gets more conservative treatment)
    is_high_demand = (
      chars.avg_abs_target_lataccel > self.high_demand_avg_threshold and
      chars.max_abs_target_lataccel > self.high_demand_max_threshold and
      not is_extreme_scenario  # Don't override extreme scenario handling
    )

    if is_high_demand:
      # High-demand mode: boost feedforward significantly, reduce PID slightly
      # Apply these as absolute multipliers on top of all other adaptations
      ff_gain *= 1.22  # Strong boost to match pid_w_ff behavior
      p_gain *= 0.92  # Let feedforward dominate
      i_gain *= 0.85  # Reduce integral to prevent interference with FF

    # Clamp gains to safe ranges
    p_gain = np.clip(p_gain, self.p_min, self.p_max)
    i_gain = np.clip(i_gain, self.i_min, self.i_max)
    d_gain = np.clip(d_gain, self.d_min, self.d_max)
    ff_gain = np.clip(ff_gain, 0.5, 1.15)  # Allow higher FF gain for high-demand scenarios

    return p_gain, i_gain, d_gain, ff_gain

  def _compute_adaptive_integral_limit(self) -> float:
    """Compute maximum integral value based on segment characteristics."""
    chars = self.segment_chars

    max_integral = self.max_integral_base

    # NEW: Limit integral in extreme difficulty scenarios
    if chars.velocity_target_product > self.velocity_target_difficulty_threshold:
      max_integral *= 0.7  # Reduce for high-difficulty scenarios

    # NEW: Limit for extreme target demands
    if chars.max_abs_target_lataccel > self.extreme_target_threshold:
      max_integral *= 0.75

    # Reduce integral limit modestly for aggressive segments
    if chars.avg_abs_target_lataccel_change > self.aggressive_target_threshold * 1.5:
      max_integral *= 0.8

    # Reduce for very high roll situations
    if chars.avg_abs_roll_lataccel > self.high_roll_threshold * 1.2:
      max_integral *= 0.85

    # Small reduction for high-speed segments
    if chars.avg_v_ego > 32.0:
      max_integral *= 0.85

    # Small increase for stable segments
    if chars.avg_v_ego < 15.0 and chars.std_v_ego < 2.0:
      max_integral *= 1.15

    return max_integral

  def _compute_error_terms(self, target_lataccel: float, current_lataccel: float, state: Any) -> Tuple[float, float]:
    """Return the proportional error and its first derivative with adaptive integral limiting."""
    error = target_lataccel - current_lataccel

    # Update integral with anti-windup
    max_integral = self._compute_adaptive_integral_limit()
    self.error_integral += error
    self.error_integral = np.clip(self.error_integral, -max_integral, max_integral)

    # Only decay integral in extremely dynamic conditions
    if self.segment_chars.avg_abs_target_lataccel_change > self.aggressive_target_threshold * 2.0:
      self.error_integral *= self.integral_decay_rate

    # Compute derivative
    error_diff = error - self.prev_error
    self.prev_error = error

    # Track recent errors for diagnostics
    self.recent_errors.append(abs(error))
    if len(self.recent_errors) > 20:
      self.recent_errors.pop(0)

    return error, error_diff

  def _feedforward(self, target_lataccel: float, state: Any) -> float:
    """
    Compute feedforward control term based on target and vehicle state.

    This anticipates the steering needed based on the desired lateral acceleration,
    compensating for roll and scaling by velocity.
    """
    # Compensate for roll-induced lateral acceleration
    steer_accel_target = target_lataccel - state.roll_lataccel

    # Velocity-dependent steering gain (higher speed = less steering needed)
    denom = max(self.steer_sat_v, max(self.minimum_velocity, state.v_ego))
    steer_command = (steer_accel_target * self.steer_factor) / denom

    # Apply sigmoid saturation for smooth limiting
    steer_command = 2 * self.steer_command_sat / (1 + np.exp(-steer_command)) - self.steer_command_sat

    return self.feedforward_gain_adaptive * steer_command

  def compute_action(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any, control_state: ControlState) -> float:
    """
    Adaptive PID + Feedforward control with online segment characteristic estimation.
    """
    # Optionally blend future targets for anticipatory control
    if self.use_future_blending and future_plan and len(future_plan.lataccel) >= 3:
      target_lataccel = self._blend_future_targets(target_lataccel, future_plan)

    # Update segment characteristics
    self._update_segment_characteristics(target_lataccel, state)

    # Compute adaptive gains (including feedforward gain)
    self.p_adaptive, self.i_adaptive, self.d_adaptive, self.feedforward_gain_adaptive = self._compute_adaptive_gains(state)

    # Compute error terms with adaptive integral limiting
    error, error_diff = self._compute_error_terms(target_lataccel, current_lataccel, state)

    # Compute PID term
    pid_term = (self.p_adaptive * error) + (self.i_adaptive * self.error_integral) + (self.d_adaptive * error_diff)

    # Compute feedforward term
    feedforward_term = self._feedforward(target_lataccel, state)

    # Combine PID and feedforward
    action = pid_term + feedforward_term

    return action

  def get_diagnostics(self) -> Dict[str, Any]:
    """Return controller diagnostics for analysis."""
    chars = self.segment_chars
    return {
      'p_adaptive': self.p_adaptive,
      'i_adaptive': self.i_adaptive,
      'd_adaptive': self.d_adaptive,
      'feedforward_gain_adaptive': self.feedforward_gain_adaptive,
      'error_integral': self.error_integral,
      'estimated_avg_v_ego': chars.avg_v_ego,
      'estimated_std_v_ego': chars.std_v_ego,
      'estimated_avg_abs_roll_lataccel': chars.avg_abs_roll_lataccel,
      'estimated_avg_abs_target_lataccel': chars.avg_abs_target_lataccel,
      'estimated_avg_abs_target_lataccel_change': chars.avg_abs_target_lataccel_change,
      'velocity_target_product': chars.velocity_target_product,
      'max_abs_target_lataccel': chars.max_abs_target_lataccel,
      'target_variability': chars.target_variability,
      'num_large_target_changes': chars.num_large_target_changes,
      'avg_recent_error': np.mean(self.recent_errors) if self.recent_errors else 0.0,
      'num_samples': chars.num_samples,
    }
