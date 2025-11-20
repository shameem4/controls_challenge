"""Neural network for adaptive PID gain tuning."""
import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING
import pickle
from pathlib import Path

if TYPE_CHECKING:
  from controllers.pid_adaptive import AdaptiveGains


class GainAdapterNetwork:
  """Simple neural network for predicting PID gains based on driving context."""

  def __init__(self, input_dim: int = 48, hidden_dims: Tuple[int, ...] = (64, 32), learning_rate: float = 0.001):
    """
    Initialize the gain adapter network.

    Args:
      input_dim: Number of input features
      hidden_dims: Tuple of hidden layer dimensions
      learning_rate: Learning rate for gradient descent
    """
    self.input_dim = input_dim
    self.hidden_dims = hidden_dims
    self.learning_rate = learning_rate

    # Initialize network parameters
    self._initialize_weights()

    # Default gain ranges (will be updated during training)
    self.gain_means = np.array([0.3, 0.07, -0.1, 0.8], dtype=np.float32)  # p, i, d, ff_gain
    self.gain_stds = np.array([0.1, 0.03, 0.05, 0.2], dtype=np.float32)

  def _initialize_weights(self) -> None:
    """Initialize network weights using Xavier initialization."""
    self.weights = []
    self.biases = []

    # Input to first hidden layer
    prev_dim = self.input_dim
    for hidden_dim in self.hidden_dims:
      w = np.random.randn(prev_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / prev_dim)
      b = np.zeros(hidden_dim, dtype=np.float32)
      self.weights.append(w)
      self.biases.append(b)
      prev_dim = hidden_dim

    # Last hidden to output (4 gains: p, i, d, feedforward_gain)
    w = np.random.randn(prev_dim, 4).astype(np.float32) * np.sqrt(2.0 / prev_dim)
    b = np.zeros(4, dtype=np.float32)
    self.weights.append(w)
    self.biases.append(b)

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

  def forward(self, x: np.ndarray) -> Tuple[np.ndarray, list]:
    """
    Forward pass through the network.

    Args:
      x: Input features (input_dim,)

    Returns:
      output: Network output (4,) - normalized gain adjustments
      activations: List of intermediate activations for backprop
    """
    activations = [x]

    # Hidden layers with ReLU
    for i in range(len(self.hidden_dims)):
      x = np.dot(x, self.weights[i]) + self.biases[i]
      x = self._relu(x)
      activations.append(x)

    # Output layer with tanh (to keep adjustments bounded)
    x = np.dot(x, self.weights[-1]) + self.biases[-1]
    x = self._tanh(x)
    activations.append(x)

    return x, activations

  def predict_gains(self, features: np.ndarray) -> 'AdaptiveGains':
    """
    Predict PID gains from features.

    Args:
      features: Input features

    Returns:
      AdaptiveGains object with predicted gains
    """
    from controllers.pid_adaptive import AdaptiveGains

    # Normalize features (simple standardization)
    features_norm = features / (np.abs(features).max() + 1e-8)

    # Forward pass
    normalized_adjustments, _ = self.forward(features_norm)

    # Convert normalized adjustments to actual gains
    gains = self.gain_means + normalized_adjustments * self.gain_stds

    # Clip gains to reasonable ranges
    p_gain = float(np.clip(gains[0], 0.1, 0.8))
    i_gain = float(np.clip(gains[1], 0.0, 0.2))
    d_gain = float(np.clip(gains[2], -0.3, 0.0))
    ff_gain = float(np.clip(gains[3], 0.3, 1.2))

    return AdaptiveGains(p=p_gain, i=i_gain, d=d_gain, feedforward_gain=ff_gain)

  def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute MSE loss.

    Args:
      predictions: Predicted normalized gain adjustments
      targets: Target normalized gain adjustments

    Returns:
      MSE loss
    """
    return float(np.mean((predictions - targets) ** 2))

  def backward(self, activations: list, target: np.ndarray, learning_rate: Optional[float] = None) -> float:
    """
    Backward pass and parameter update.

    Args:
      activations: List of activations from forward pass
      target: Target normalized gain adjustments
      learning_rate: Learning rate (uses self.learning_rate if None)

    Returns:
      loss: MSE loss
    """
    if learning_rate is None:
      learning_rate = self.learning_rate

    # Compute loss
    output = activations[-1]
    loss = self.compute_loss(output, target)

    # Output layer gradient
    delta = 2 * (output - target) * self._tanh_derivative(output)

    # Backpropagate through layers
    for i in range(len(self.weights) - 1, -1, -1):
      # Compute gradients
      grad_w = np.outer(activations[i], delta)
      grad_b = delta

      # Gradient clipping to prevent explosionexplosion
      max_grad_norm = 1.0
      grad_w = np.clip(grad_w, -max_grad_norm, max_grad_norm)
      grad_b = np.clip(grad_b, -max_grad_norm, max_grad_norm)

      # Update weights
      self.weights[i] -= learning_rate * grad_w
      self.biases[i] -= learning_rate * grad_b

      # Check for NaN/Inf and reset if needed
      if np.any(np.isnan(self.weights[i])) or np.any(np.isinf(self.weights[i])):
        print(f"Warning: NaN/Inf detected in weights[{i}], resetting layer")
        prev_dim = self.weights[i].shape[0]
        curr_dim = self.weights[i].shape[1]
        self.weights[i] = np.random.randn(prev_dim, curr_dim).astype(np.float32) * np.sqrt(2.0 / prev_dim) * 0.1
        self.biases[i] = np.zeros(curr_dim, dtype=np.float32)

      # Propagate delta to previous layer
      if i > 0:
        delta = np.dot(self.weights[i], delta) * self._relu_derivative(activations[i])
        # Clip delta to prevent explosion in earlier layers
        delta = np.clip(delta, -max_grad_norm, max_grad_norm)

    return loss

  def update_gain_statistics(self, gains_history: list) -> None:
    """
    Update gain mean and std based on observed gains.

    Args:
      gains_history: List of (p, i, d, ff_gain) tuples
    """
    if len(gains_history) > 0:
      gains_array = np.array(gains_history, dtype=np.float32)
      self.gain_means = np.mean(gains_array, axis=0)
      self.gain_stds = np.std(gains_array, axis=0) + 1e-6

  def save(self, path: str) -> None:
    """Save network to file."""
    state = {
      'weights': self.weights,
      'biases': self.biases,
      'gain_means': self.gain_means,
      'gain_stds': self.gain_stds,
      'input_dim': self.input_dim,
      'hidden_dims': self.hidden_dims,
      'learning_rate': self.learning_rate,
    }
    with open(path, 'wb') as f:
      pickle.dump(state, f)

  @classmethod
  def load(cls, path: str) -> 'GainAdapterNetwork':
    """Load network from file."""
    with open(path, 'rb') as f:
      state = pickle.load(f)

    network = cls(
      input_dim=state['input_dim'],
      hidden_dims=state['hidden_dims'],
      learning_rate=state['learning_rate']
    )
    network.weights = state['weights']
    network.biases = state['biases']
    network.gain_means = state['gain_means']
    network.gain_stds = state['gain_stds']

    return network


class ReplayBuffer:
  """Experience replay buffer for training."""

  def __init__(self, capacity: int = 10000):
    self.capacity = capacity
    self.buffer = []
    self.position = 0

  def push(self, experience: Tuple[np.ndarray, np.ndarray, float]) -> None:
    """
    Add experience to buffer.

    Args:
      experience: (features, gains, cost) tuple
    """
    if len(self.buffer) < self.capacity:
      self.buffer.append(experience)
    else:
      self.buffer[self.position] = experience
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size: int) -> list:
    """Sample random batch from buffer."""
    indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
    return [self.buffer[i] for i in indices]

  def __len__(self) -> int:
    return len(self.buffer)

  def get_best_experiences(self, n: int) -> list:
    """Get n experiences with lowest cost."""
    sorted_buffer = sorted(self.buffer, key=lambda x: x[2])  # Sort by cost
    return sorted_buffer[:min(n, len(sorted_buffer))]
