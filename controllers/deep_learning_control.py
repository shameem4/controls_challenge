"""Deep learning controller using unsupervised learning with convolutional neural network."""
from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np
from pathlib import Path

from . import BaseController, ControlState


@dataclass
class Controller(BaseController):
  """
  Deep learning controller with unsupervised learning.

  Uses a convolutional neural network that:
  - Takes history of control variables and future plan as input
  - Outputs control action and predicted next lat_accel
  - Learns from actual lat_accel feedback from simulator (unsupervised)
  """

  # Network parameters
  model_path: str = "models/deep_learning_control.pkl"
  history_length: int = 20
  future_length: int = 10

  # Online learning
  enable_online_learning: bool = False
  online_learning_rate: float = 0.00001

  # Internal state
  network: Optional[Any] = field(default=None, init=False)
  error_history: List[float] = field(default_factory=list, init=False)
  lataccel_history: List[float] = field(default_factory=list, init=False)
  target_history: List[float] = field(default_factory=list, init=False)
  v_ego_history: List[float] = field(default_factory=list, init=False)
  a_ego_history: List[float] = field(default_factory=list, init=False)
  roll_lataccel_history: List[float] = field(default_factory=list, init=False)
  action_history: List[float] = field(default_factory=list, init=False)

  # For online learning
  last_prediction: Optional[float] = field(default=None, init=False)
  last_activations: Optional[List] = field(default=None, init=False)

  # Warmup parameters
  warmup_steps: int = 10
  control_steps: int = field(default=0, init=False)

  # Diagnostics
  total_prediction_error: float = field(default=0.0, init=False)
  prediction_count: int = field(default=0, init=False)

  def __post_init__(self):
    """Load the trained network."""
    self._load_network()

  def _load_network(self) -> None:
    """Load the trained deep learning network."""
    try:
      from deep_learning_network import DeepLearningControlNetwork

      model_file = Path(self.model_path)
      if model_file.exists():
        self.network = DeepLearningControlNetwork.load(str(model_file))
        print(f"Loaded deep learning control network from {self.model_path}")
      else:
        print(f"Warning: Model file {self.model_path} not found. Creating new network.")
        # Create default network
        self.network = DeepLearningControlNetwork(
          history_length=self.history_length,
          future_length=self.future_length,
          num_channels=7,
          conv_filters=(16, 32, 64),
          kernel_size=3,
          fc_dims=(128, 64),
          learning_rate=self.online_learning_rate
        )
    except Exception as e:
      print(f"Error loading network: {e}")
      print("Deep learning controller will not function properly.")
      self.network = None

  def reset(self) -> None:
    """Reset controller state."""
    self.error_history.clear()
    self.lataccel_history.clear()
    self.target_history.clear()
    self.v_ego_history.clear()
    self.a_ego_history.clear()
    self.roll_lataccel_history.clear()
    self.action_history.clear()
    self.control_steps = 0
    self.last_prediction = None
    self.last_activations = None
    self.total_prediction_error = 0.0
    self.prediction_count = 0

  def _prepare_history(self) -> np.ndarray:
    """
    Prepare history matrix from individual histories.

    Returns:
      history: (history_length, num_channels) array
    """
    def pad_history(hist: List[float], length: int) -> np.ndarray:
      if len(hist) >= length:
        return np.array(hist[-length:], dtype=np.float32)
      else:
        padded = np.zeros(length, dtype=np.float32)
        if len(hist) > 0:
          padded[-len(hist):] = hist
        return padded

    error = pad_history(self.error_history, self.history_length)
    current_lataccel = pad_history(self.lataccel_history, self.history_length)
    target_lataccel = pad_history(self.target_history, self.history_length)
    v_ego = pad_history(self.v_ego_history, self.history_length)
    a_ego = pad_history(self.a_ego_history, self.history_length)
    roll_lataccel = pad_history(self.roll_lataccel_history, self.history_length)
    action = pad_history(self.action_history, self.history_length)

    # Stack into (time, channels) format
    history = np.stack([
      error,
      current_lataccel,
      target_lataccel,
      v_ego,
      a_ego,
      roll_lataccel,
      action
    ], axis=1)

    return history

  def _prepare_future_plan(self, future_plan: Any) -> np.ndarray:
    """
    Prepare future plan matrix.

    Returns:
      future: (future_length, 4) array
    """
    def pad_to_length(data: List[float], length: int) -> np.ndarray:
      if not data:
        return np.zeros(length, dtype=np.float32)
      arr = np.array(data, dtype=np.float32)
      if len(arr) >= length:
        return arr[:length]
      else:
        padded = np.zeros(length, dtype=np.float32)
        padded[:len(arr)] = arr
        return padded

    if future_plan is None:
      return np.zeros((self.future_length, 4), dtype=np.float32)

    lataccel = pad_to_length(future_plan.lataccel if hasattr(future_plan, 'lataccel') else [], self.future_length)
    roll_lataccel = pad_to_length(future_plan.roll_lataccel if hasattr(future_plan, 'roll_lataccel') else [], self.future_length)
    v_ego = pad_to_length(future_plan.v_ego if hasattr(future_plan, 'v_ego') else [], self.future_length)
    a_ego = pad_to_length(future_plan.a_ego if hasattr(future_plan, 'a_ego') else [], self.future_length)

    future = np.stack([lataccel, roll_lataccel, v_ego, a_ego], axis=1)
    return future

  def _update_history(
    self,
    error: float,
    current_lataccel: float,
    target_lataccel: float,
    v_ego: float,
    a_ego: float,
    roll_lataccel: float,
    action: float
  ) -> None:
    """Update history buffers."""
    self.error_history.append(error)
    self.lataccel_history.append(current_lataccel)
    self.target_history.append(target_lataccel)
    self.v_ego_history.append(v_ego)
    self.a_ego_history.append(a_ego)
    self.roll_lataccel_history.append(roll_lataccel)
    self.action_history.append(action)

    # Keep only recent history
    max_history = self.history_length * 2  # Keep a bit extra
    if len(self.error_history) > max_history:
      self.error_history.pop(0)
      self.lataccel_history.pop(0)
      self.target_history.pop(0)
      self.v_ego_history.pop(0)
      self.a_ego_history.pop(0)
      self.roll_lataccel_history.pop(0)
      self.action_history.pop(0)

  def compute_action(
    self,
    target_lataccel: float,
    current_lataccel: float,
    state: Any,
    future_plan: Any,
    control_state: ControlState
  ) -> float:
    """
    Compute control action using deep learning network.

    Args:
      target_lataccel: Target lateral acceleration
      current_lataccel: Current lateral acceleration
      state: Current vehicle state
      future_plan: Future trajectory plan
      control_state: Control state metadata

    Returns:
      action: Control action (steering command)
    """
    if self.network is None:
      # Fallback to simple proportional control
      error = target_lataccel - current_lataccel
      return error * 0.3

    error = target_lataccel - current_lataccel

    # Warmup phase: use simple control and build history
    if self.control_steps < self.warmup_steps:
      action = error * 0.3  # Simple P controller
      self._update_history(
        error,
        current_lataccel,
        target_lataccel,
        state.v_ego,
        state.a_ego,
        state.roll_lataccel,
        action
      )
      self.control_steps += 1
      return float(action)

    # Prepare inputs for network
    history = self._prepare_history()
    future = self._prepare_future_plan(future_plan)

    # Get network prediction
    try:
      # For inference, we use predict which handles normalization
      action, predicted_lataccel = self.network.predict(history, future)

      # Store for online learning
      if self.enable_online_learning:
        # Get activations for potential update
        history_norm = (history - self.network.input_mean) / (self.network.input_std + 1e-8)
        future_norm = (future - self.network.future_mean) / (self.network.future_std + 1e-8)
        _, pred_lataccel_norm, activations = self.network.forward(history_norm, future_norm)

        self.last_prediction = predicted_lataccel
        self.last_activations = activations

      # Record diagnostic
      control_state.record('predicted_lataccel', predicted_lataccel)
      control_state.record('action', action)

    except Exception as e:
      print(f"Error in network forward pass: {e}")
      # Fallback to proportional control
      action = error * 0.3
      predicted_lataccel = None

    # Update history
    self._update_history(
      error,
      current_lataccel,
      target_lataccel,
      state.v_ego,
      state.a_ego,
      state.roll_lataccel,
      action
    )

    self.control_steps += 1

    return float(action)

  def on_simulation_update(
    self,
    predicted_lataccel: float,
    control_state: ControlState,
    step_metrics: Optional[dict] = None
  ) -> None:
    """
    Hook called after simulator produces next lat_accel.

    This is where we perform online learning using the actual lat_accel as supervision.

    Args:
      predicted_lataccel: Actual lat_accel from simulator
      control_state: Control state metadata
      step_metrics: Per-step metrics from simulator
    """
    # Skip during warmup
    if self.control_steps <= self.warmup_steps:
      return

    # Skip if we don't have a prediction to compare against
    if self.last_prediction is None or self.last_activations is None:
      return

    # Skip if network not available or online learning disabled
    if self.network is None or not self.enable_online_learning:
      # Still track prediction error for diagnostics
      if self.last_prediction is not None:
        pred_error = abs(self.last_prediction - predicted_lataccel)
        self.total_prediction_error += pred_error
        self.prediction_count += 1
      return

    # Compute prediction error
    pred_error = abs(self.last_prediction - predicted_lataccel)
    self.total_prediction_error += pred_error
    self.prediction_count += 1

    # Perform online learning
    try:
      loss, prediction_loss = self.network.backward_and_update(
        self.last_activations,
        predicted_lataccel,
        learning_rate=self.online_learning_rate
      )

      # Record for diagnostics
      control_state.record('dl_loss', loss)
      control_state.record('dl_prediction_error', pred_error)

    except Exception as e:
      print(f"Error in online learning update: {e}")

    # Clear stored prediction and activations
    self.last_prediction = None
    self.last_activations = None

  def get_diagnostics(self) -> dict:
    """Return diagnostic information."""
    diagnostics = {}

    if self.prediction_count > 0:
      avg_pred_error = self.total_prediction_error / self.prediction_count
      diagnostics['avg_prediction_error'] = avg_pred_error

    diagnostics['control_steps'] = self.control_steps
    diagnostics['online_learning_enabled'] = self.enable_online_learning

    if self.network is not None:
      diagnostics['network_train_steps'] = self.network.train_steps

    return diagnostics
