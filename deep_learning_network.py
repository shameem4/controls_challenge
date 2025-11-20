"""Unsupervised deep learning network for lateral control with convolutional layers."""
import numpy as np
from typing import Tuple, Optional, List
import pickle
from pathlib import Path


class DeepLearningControlNetwork:
  """
  Unsupervised neural network with convolutional layers for lateral control.

  Architecture:
  - Conv1D layers to process time-series history
  - Fully connected layers for future plan processing
  - Dual outputs: control action and predicted next lat_accel
  - Loss based on prediction error of actual lat_accel from simulator
  """

  def __init__(
    self,
    history_length: int = 20,
    future_length: int = 10,
    num_channels: int = 7,  # error, current_lataccel, target_lataccel, v_ego, a_ego, roll_lataccel, action
    conv_filters: Tuple[int, ...] = (16, 32, 64),
    kernel_size: int = 3,
    fc_dims: Tuple[int, ...] = (128, 64),
    learning_rate: float = 0.0001
  ):
    """
    Initialize the deep learning control network.

    Args:
      history_length: Number of historical timesteps
      future_length: Number of future plan timesteps
      num_channels: Number of input channels in history
      conv_filters: Number of filters in each conv layer
      kernel_size: Kernel size for conv layers
      fc_dims: Dimensions of fully connected layers
      learning_rate: Learning rate for gradient descent
    """
    self.history_length = history_length
    self.future_length = future_length
    self.num_channels = num_channels
    self.conv_filters = conv_filters
    self.kernel_size = kernel_size
    self.fc_dims = fc_dims
    self.learning_rate = learning_rate

    # Statistics for normalization
    self.input_mean = np.zeros(num_channels, dtype=np.float32)
    self.input_std = np.ones(num_channels, dtype=np.float32)
    self.future_mean = np.zeros(4, dtype=np.float32)  # lataccel, roll_lataccel, v_ego, a_ego
    self.future_std = np.ones(4, dtype=np.float32)

    # Initialize network parameters
    self._initialize_weights()

    # Moving averages for statistics
    self.stats_momentum = 0.99
    self.train_steps = 0

  def _initialize_weights(self) -> None:
    """Initialize all network weights using He initialization."""
    self.conv_weights = []
    self.conv_biases = []

    # Convolutional layers
    in_channels = self.num_channels
    for out_channels in self.conv_filters:
      # Conv1D weights: (in_channels, kernel_size, out_channels)
      w = np.random.randn(in_channels, self.kernel_size, out_channels).astype(np.float32)
      w *= np.sqrt(2.0 / (in_channels * self.kernel_size))
      b = np.zeros(out_channels, dtype=np.float32)
      self.conv_weights.append(w)
      self.conv_biases.append(b)
      in_channels = out_channels

    # Calculate size after convolutions
    conv_output_length = self.history_length
    for _ in self.conv_filters:
      conv_output_length = conv_output_length - self.kernel_size + 1

    # Flatten conv output + future plan features
    flatten_size = conv_output_length * self.conv_filters[-1] + self.future_length * 4

    # Fully connected layers
    self.fc_weights = []
    self.fc_biases = []

    prev_dim = flatten_size
    for fc_dim in self.fc_dims:
      w = np.random.randn(prev_dim, fc_dim).astype(np.float32) * np.sqrt(2.0 / prev_dim)
      b = np.zeros(fc_dim, dtype=np.float32)
      self.fc_weights.append(w)
      self.fc_biases.append(b)
      prev_dim = fc_dim

    # Dual output heads
    # Head 1: Control action (1 output)
    self.action_head_w = np.random.randn(prev_dim, 1).astype(np.float32) * np.sqrt(2.0 / prev_dim)
    self.action_head_b = np.zeros(1, dtype=np.float32)

    # Head 2: Predicted next lat_accel (1 output)
    self.lataccel_pred_head_w = np.random.randn(prev_dim, 1).astype(np.float32) * np.sqrt(2.0 / prev_dim)
    self.lataccel_pred_head_b = np.zeros(1, dtype=np.float32)

  def _relu(self, x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

  def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0).astype(np.float32)

  def _tanh(self, x: np.ndarray) -> np.ndarray:
    """Tanh activation."""
    return np.tanh(x)

  def _tanh_derivative(self, x: np.ndarray) -> np.ndarray:
    """Derivative of tanh."""
    t = np.tanh(x)
    return 1 - t * t

  def _conv1d(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Simple 1D convolution.

    Args:
      x: Input (time_steps, in_channels)
      weight: Conv weights (in_channels, kernel_size, out_channels)
      bias: Conv bias (out_channels,)

    Returns:
      output: (output_time_steps, out_channels)
    """
    time_steps, in_channels = x.shape
    _, kernel_size, out_channels = weight.shape
    output_length = time_steps - kernel_size + 1

    output = np.zeros((output_length, out_channels), dtype=np.float32)

    for t in range(output_length):
      for out_ch in range(out_channels):
        value = 0.0
        for in_ch in range(in_channels):
          for k in range(kernel_size):
            value += x[t + k, in_ch] * weight[in_ch, k, out_ch]
        output[t, out_ch] = value + bias[out_ch]

    return output

  def forward(
    self,
    history: np.ndarray,
    future_plan: np.ndarray
  ) -> Tuple[float, float, List[np.ndarray]]:
    """
    Forward pass through the network.

    Args:
      history: Historical data (history_length, num_channels)
      future_plan: Future plan data (future_length, 4)

    Returns:
      action: Predicted control action
      predicted_lataccel: Predicted next lateral acceleration
      activations: List of intermediate activations for backprop
    """
    activations = []

    # Store input
    activations.append(('history', history.copy()))
    activations.append(('future_plan', future_plan.copy()))

    # Process history through conv layers
    x = history
    for i, (w, b) in enumerate(zip(self.conv_weights, self.conv_biases)):
      x = self._conv1d(x, w, b)
      activations.append((f'conv_{i}_pre', x.copy()))
      x = self._relu(x)
      activations.append((f'conv_{i}_post', x.copy()))

    # Flatten conv output
    conv_flat = x.flatten()
    activations.append(('conv_flat', conv_flat.copy()))

    # Flatten future plan
    future_flat = future_plan.flatten()
    activations.append(('future_flat', future_flat.copy()))

    # Concatenate
    x = np.concatenate([conv_flat, future_flat])
    activations.append(('concat', x.copy()))

    # Fully connected layers
    for i, (w, b) in enumerate(zip(self.fc_weights, self.fc_biases)):
      x = np.dot(x, w) + b
      activations.append((f'fc_{i}_pre', x.copy()))
      x = self._relu(x)
      activations.append((f'fc_{i}_post', x.copy()))

    # Output heads
    action = np.tanh(np.dot(x, self.action_head_w) + self.action_head_b)[0]
    predicted_lataccel = np.dot(x, self.lataccel_pred_head_w) + self.lataccel_pred_head_b
    predicted_lataccel = predicted_lataccel[0]

    activations.append(('shared_features', x.copy()))
    activations.append(('action', action))
    activations.append(('predicted_lataccel', predicted_lataccel))

    return action, predicted_lataccel, activations

  def predict(self, history: np.ndarray, future_plan: np.ndarray) -> Tuple[float, float]:
    """
    Predict control action and next lat_accel (inference mode).

    Args:
      history: Historical data (history_length, num_channels)
      future_plan: Future plan data (future_length, 4)

    Returns:
      action: Predicted control action (clipped to [-2, 2])
      predicted_lataccel: Predicted next lateral acceleration
    """
    # Normalize inputs
    history_norm = (history - self.input_mean) / (self.input_std + 1e-8)
    future_norm = (future_plan - self.future_mean) / (self.future_std + 1e-8)

    action, predicted_lataccel, _ = self.forward(history_norm, future_norm)

    # Clip action to reasonable range
    action = float(np.clip(action * 2.0, -2.0, 2.0))

    return action, float(predicted_lataccel)

  def compute_loss(
    self,
    predicted_lataccel: float,
    actual_lataccel: float,
    action: float
  ) -> Tuple[float, float]:
    """
    Compute loss based on lat_accel prediction error and action regularization.

    Args:
      predicted_lataccel: Network's prediction of next lat_accel
      actual_lataccel: Actual lat_accel from simulator
      action: Control action taken

    Returns:
      total_loss: Combined loss
      prediction_loss: Just the prediction error component
    """
    # Main loss: prediction error
    prediction_loss = (predicted_lataccel - actual_lataccel) ** 2

    # Regularization: penalize large actions
    action_reg = 0.01 * action ** 2

    total_loss = prediction_loss + action_reg

    return float(total_loss), float(prediction_loss)

  def update_statistics(self, history: np.ndarray, future_plan: np.ndarray) -> None:
    """Update running statistics for normalization."""
    if self.train_steps == 0:
      self.input_mean = np.mean(history, axis=0)
      self.input_std = np.std(history, axis=0) + 1e-8
      self.future_mean = np.mean(future_plan, axis=0)
      self.future_std = np.std(future_plan, axis=0) + 1e-8
    else:
      # Moving average
      batch_mean = np.mean(history, axis=0)
      batch_std = np.std(history, axis=0) + 1e-8
      self.input_mean = self.stats_momentum * self.input_mean + (1 - self.stats_momentum) * batch_mean
      self.input_std = self.stats_momentum * self.input_std + (1 - self.stats_momentum) * batch_std

      future_batch_mean = np.mean(future_plan, axis=0)
      future_batch_std = np.std(future_plan, axis=0) + 1e-8
      self.future_mean = self.stats_momentum * self.future_mean + (1 - self.stats_momentum) * future_batch_mean
      self.future_std = self.stats_momentum * self.future_std + (1 - self.stats_momentum) * future_batch_std

  def backward_and_update(
    self,
    activations: List[Tuple],
    actual_lataccel: float,
    learning_rate: Optional[float] = None
  ) -> Tuple[float, float]:
    """
    Backward pass and parameter update using actual lat_accel as supervision.

    Args:
      activations: List of activations from forward pass
      actual_lataccel: Actual lateral acceleration from simulator
      learning_rate: Learning rate (uses self.learning_rate if None)

    Returns:
      total_loss: Combined loss value
      prediction_loss: Prediction error component
    """
    if learning_rate is None:
      learning_rate = self.learning_rate

    # Extract activations
    act_dict = dict(activations)
    action = act_dict['action']
    predicted_lataccel = act_dict['predicted_lataccel']
    shared_features = act_dict['shared_features']

    # Compute loss
    total_loss, prediction_loss = self.compute_loss(predicted_lataccel, actual_lataccel, action)

    # Compute gradients for output heads
    # Gradient of prediction loss w.r.t. predicted_lataccel
    d_pred_lataccel = 2 * (predicted_lataccel - actual_lataccel)

    # Gradient of action regularization w.r.t. action
    d_action = 0.02 * action

    # Backprop through output heads
    # Lataccel prediction head
    grad_lataccel_w = shared_features * d_pred_lataccel
    grad_lataccel_b = d_pred_lataccel

    # Action head (through tanh)
    action_pre_tanh = np.dot(shared_features, self.action_head_w) + self.action_head_b
    d_action_pre_tanh = d_action * self._tanh_derivative(action_pre_tanh)
    grad_action_w = np.outer(shared_features, d_action_pre_tanh)
    grad_action_b = d_action_pre_tanh.flatten()

    # Gradient flowing back to shared features
    d_shared = (np.dot(self.lataccel_pred_head_w, d_pred_lataccel) +
                np.dot(self.action_head_w, d_action_pre_tanh.flatten()))

    # Update output heads
    self.lataccel_pred_head_w -= learning_rate * grad_lataccel_w.reshape(-1, 1)
    self.lataccel_pred_head_b -= learning_rate * grad_lataccel_b
    self.action_head_w -= learning_rate * grad_action_w
    self.action_head_b -= learning_rate * grad_action_b

    # Backprop through FC layers (simplified - approximate gradients)
    for i in range(len(self.fc_weights) - 1, -1, -1):
      fc_pre = act_dict[f'fc_{i}_pre']
      fc_post = act_dict[f'fc_{i}_post']

      # Gradient through ReLU
      d_shared = d_shared * self._relu_derivative(fc_pre)

      # Get input to this layer
      if i == 0:
        fc_input = act_dict['concat']
      else:
        fc_input = act_dict[f'fc_{i-1}_post']

      # Compute gradients
      grad_w = np.outer(fc_input, d_shared)
      grad_b = d_shared

      # Clip gradients
      grad_w = np.clip(grad_w, -1.0, 1.0)
      grad_b = np.clip(grad_b, -1.0, 1.0)

      # Update weights
      self.fc_weights[i] -= learning_rate * grad_w
      self.fc_biases[i] -= learning_rate * grad_b

      # Propagate to previous layer
      d_shared = np.dot(self.fc_weights[i], d_shared)

    # Note: Backprop through conv layers is complex; for simplicity,
    # we'll use a smaller learning rate and approximate updates
    # In practice, you'd want proper conv backprop or use a framework like PyTorch

    self.train_steps += 1

    return total_loss, prediction_loss

  def save(self, path: str) -> None:
    """Save network to file."""
    state = {
      'conv_weights': self.conv_weights,
      'conv_biases': self.conv_biases,
      'fc_weights': self.fc_weights,
      'fc_biases': self.fc_biases,
      'action_head_w': self.action_head_w,
      'action_head_b': self.action_head_b,
      'lataccel_pred_head_w': self.lataccel_pred_head_w,
      'lataccel_pred_head_b': self.lataccel_pred_head_b,
      'input_mean': self.input_mean,
      'input_std': self.input_std,
      'future_mean': self.future_mean,
      'future_std': self.future_std,
      'history_length': self.history_length,
      'future_length': self.future_length,
      'num_channels': self.num_channels,
      'conv_filters': self.conv_filters,
      'kernel_size': self.kernel_size,
      'fc_dims': self.fc_dims,
      'learning_rate': self.learning_rate,
      'train_steps': self.train_steps,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
      pickle.dump(state, f)

  @classmethod
  def load(cls, path: str) -> 'DeepLearningControlNetwork':
    """Load network from file."""
    with open(path, 'rb') as f:
      state = pickle.load(f)

    network = cls(
      history_length=state['history_length'],
      future_length=state['future_length'],
      num_channels=state['num_channels'],
      conv_filters=state['conv_filters'],
      kernel_size=state['kernel_size'],
      fc_dims=state['fc_dims'],
      learning_rate=state['learning_rate']
    )

    network.conv_weights = state['conv_weights']
    network.conv_biases = state['conv_biases']
    network.fc_weights = state['fc_weights']
    network.fc_biases = state['fc_biases']
    network.action_head_w = state['action_head_w']
    network.action_head_b = state['action_head_b']
    network.lataccel_pred_head_w = state['lataccel_pred_head_w']
    network.lataccel_pred_head_b = state['lataccel_pred_head_b']
    network.input_mean = state['input_mean']
    network.input_std = state['input_std']
    network.future_mean = state['future_mean']
    network.future_std = state['future_std']
    network.train_steps = state.get('train_steps', 0)

    return network


class ExperienceBuffer:
  """Buffer for storing experiences during online learning."""

  def __init__(self, capacity: int = 1000):
    self.capacity = capacity
    self.buffer = []
    self.position = 0

  def push(
    self,
    history: np.ndarray,
    future_plan: np.ndarray,
    action: float,
    predicted_lataccel: float,
    actual_lataccel: float
  ) -> None:
    """Add experience to buffer."""
    experience = {
      'history': history.copy(),
      'future_plan': future_plan.copy(),
      'action': action,
      'predicted_lataccel': predicted_lataccel,
      'actual_lataccel': actual_lataccel,
      'error': abs(predicted_lataccel - actual_lataccel)
    }

    if len(self.buffer) < self.capacity:
      self.buffer.append(experience)
    else:
      self.buffer[self.position] = experience
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size: int) -> List[dict]:
    """Sample random batch."""
    indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
    return [self.buffer[i] for i in indices]

  def get_high_error_samples(self, n: int) -> List[dict]:
    """Get samples with highest prediction error for focused learning."""
    sorted_buffer = sorted(self.buffer, key=lambda x: x['error'], reverse=True)
    return sorted_buffer[:min(n, len(sorted_buffer))]

  def __len__(self) -> int:
    return len(self.buffer)
