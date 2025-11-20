"""Training script for unsupervised deep learning controller."""
import sys
from pathlib import Path
import numpy as np
from typing import List, Tuple
import json

from deep_learning_network import DeepLearningControlNetwork, ExperienceBuffer
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
from tinyphysics_core import State, FuturePlan


class DeepLearningControlTrainer:
  """Trainer for unsupervised deep learning control using online learning."""

  def __init__(
    self,
    network: DeepLearningControlNetwork,
    data_path: str = "data",
    model_save_path: str = "models/deep_learning_control.pkl",
    batch_size: int = 32,
    buffer_capacity: int = 5000
  ):
    self.network = network
    self.data_path = Path(data_path)
    self.model_save_path = model_save_path
    self.batch_size = batch_size
    self.experience_buffer = ExperienceBuffer(capacity=buffer_capacity)

    # Load simulator
    self.model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

    # Training metrics
    self.episode_losses = []
    self.prediction_errors = []

  def _prepare_history(
    self,
    error_history: List[float],
    lataccel_history: List[float],
    target_history: List[float],
    v_ego_history: List[float],
    a_ego_history: List[float],
    roll_lataccel_history: List[float],
    action_history: List[float]
  ) -> np.ndarray:
    """
    Prepare history matrix from individual histories.

    Returns:
      history: (history_length, num_channels) array
    """
    history_length = self.network.history_length

    # Pad histories if too short
    def pad_history(hist: List[float], length: int) -> np.ndarray:
      if len(hist) >= length:
        return np.array(hist[-length:], dtype=np.float32)
      else:
        padded = np.zeros(length, dtype=np.float32)
        padded[-len(hist):] = hist
        return padded

    error = pad_history(error_history, history_length)
    current_lataccel = pad_history(lataccel_history, history_length)
    target_lataccel = pad_history(target_history, history_length)
    v_ego = pad_history(v_ego_history, history_length)
    a_ego = pad_history(a_ego_history, history_length)
    roll_lataccel = pad_history(roll_lataccel_history, history_length)
    action = pad_history(action_history, history_length)

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

  def _prepare_future_plan(self, future_plan: FuturePlan) -> np.ndarray:
    """
    Prepare future plan matrix.

    Returns:
      future: (future_length, 4) array
    """
    future_length = self.network.future_length

    def pad_to_length(data: List[float], length: int) -> np.ndarray:
      arr = np.array(data, dtype=np.float32)
      if len(arr) >= length:
        return arr[:length]
      else:
        padded = np.zeros(length, dtype=np.float32)
        padded[:len(arr)] = arr
        return padded

    lataccel = pad_to_length(future_plan.lataccel, future_length)
    roll_lataccel = pad_to_length(future_plan.roll_lataccel, future_length)
    v_ego = pad_to_length(future_plan.v_ego, future_length)
    a_ego = pad_to_length(future_plan.a_ego, future_length)

    future = np.stack([lataccel, roll_lataccel, v_ego, a_ego], axis=1)
    return future

  def collect_episode(self, segment_path: str) -> Tuple[float, float]:
    """
    Collect experience from one episode with online learning.

    Returns:
      avg_loss: Average loss for the episode
      avg_prediction_error: Average prediction error
    """
    # Load segment data
    from tinyphysics import load_segment
    segment = load_segment(segment_path)

    # Initialize histories
    error_history = []
    lataccel_history = []
    target_history = []
    v_ego_history = []
    a_ego_history = []
    roll_lataccel_history = []
    action_history = []

    # Initialize simulator state
    current_lataccel = 0.0
    episode_losses = []
    episode_pred_errors = []

    # Warmup phase
    warmup_steps = min(self.network.history_length, 10)

    for step_idx, (state_data, target_lataccel_data, future_plan_data) in enumerate(segment):
      # Extract state
      state = State(
        roll_lataccel=state_data['roll_lataccel'],
        v_ego=state_data['v_ego'],
        a_ego=state_data['a_ego']
      )

      # Extract future plan
      future_plan = FuturePlan(
        lataccel=future_plan_data['lataccel'],
        roll_lataccel=future_plan_data['roll_lataccel'],
        v_ego=future_plan_data['v_ego'],
        a_ego=future_plan_data['a_ego']
      )

      target_lataccel = target_lataccel_data

      # Compute error
      error = target_lataccel - current_lataccel

      # Warmup with simple action
      if step_idx < warmup_steps:
        action = error * 0.3  # Simple proportional control
        predicted_lataccel = current_lataccel + action * 0.5  # Rough estimate
      else:
        # Prepare inputs for network
        history = self._prepare_history(
          error_history,
          lataccel_history,
          target_history,
          v_ego_history,
          a_ego_history,
          roll_lataccel_history,
          action_history
        )
        future = self._prepare_future_plan(future_plan)

        # Get network prediction
        action, predicted_lataccel = self.network.predict(history, future)

      # Update histories
      error_history.append(error)
      lataccel_history.append(current_lataccel)
      target_history.append(target_lataccel)
      v_ego_history.append(state.v_ego)
      a_ego_history.append(state.a_ego)
      roll_lataccel_history.append(state.roll_lataccel)
      action_history.append(action)

      # Simulate the action
      actual_lataccel = self.model.step(action)

      # After warmup, train on the prediction error
      if step_idx >= warmup_steps:
        # Store experience
        self.experience_buffer.push(
          history,
          future,
          action,
          predicted_lataccel,
          actual_lataccel
        )

        # Online learning: update immediately with current experience
        history_norm = (history - self.network.input_mean) / (self.network.input_std + 1e-8)
        future_norm = (future - self.network.future_mean) / (self.network.future_std + 1e-8)

        _, _, activations = self.network.forward(history_norm, future_norm)
        loss, pred_error = self.network.backward_and_update(activations, actual_lataccel)

        episode_losses.append(loss)
        episode_pred_errors.append(pred_error)

        # Periodically train on replay buffer
        if step_idx % 10 == 0 and len(self.experience_buffer) >= self.batch_size:
          batch = self.experience_buffer.sample(self.batch_size)
          for exp in batch[:5]:  # Train on subset to avoid overfitting
            h = (exp['history'] - self.network.input_mean) / (self.network.input_std + 1e-8)
            f = (exp['future_plan'] - self.network.future_mean) / (self.network.future_std + 1e-8)
            _, _, acts = self.network.forward(h, f)
            self.network.backward_and_update(acts, exp['actual_lataccel'], learning_rate=self.network.learning_rate * 0.5)

      # Update current lataccel for next iteration
      current_lataccel = actual_lataccel

      # Update statistics periodically
      if step_idx % 20 == 0 and step_idx > 0:
        recent_histories = [self._prepare_history(
          error_history[-20:],
          lataccel_history[-20:],
          target_history[-20:],
          v_ego_history[-20:],
          a_ego_history[-20:],
          roll_lataccel_history[-20:],
          action_history[-20:]
        )]
        recent_futures = [self._prepare_future_plan(future_plan)]

        if len(recent_histories) > 0:
          self.network.update_statistics(
            np.array(recent_histories),
            np.array(recent_futures)
          )

    avg_loss = np.mean(episode_losses) if episode_losses else 0.0
    avg_pred_error = np.mean(episode_pred_errors) if episode_pred_errors else 0.0

    return float(avg_loss), float(avg_pred_error)

  def train(self, num_episodes: int = 100, save_interval: int = 10) -> None:
    """
    Train the network over multiple episodes.

    Args:
      num_episodes: Number of episodes to train
      save_interval: Save model every N episodes
    """
    # Get list of training segments
    segment_files = sorted(self.data_path.glob("*.csv"))

    if len(segment_files) == 0:
      print(f"No training data found in {self.data_path}")
      return

    print(f"Found {len(segment_files)} training segments")
    print(f"Network architecture: {self.network.conv_filters} conv filters, {self.network.fc_dims} FC dims")
    print(f"History length: {self.network.history_length}, Future length: {self.network.future_length}")
    print()

    training_log = []

    for episode in range(num_episodes):
      # Cycle through segments
      segment_idx = episode % len(segment_files)
      segment_path = str(segment_files[segment_idx])

      print(f"Episode {episode + 1}/{num_episodes} - Segment: {segment_files[segment_idx].name}")

      try:
        avg_loss, avg_pred_error = self.collect_episode(segment_path)

        self.episode_losses.append(avg_loss)
        self.prediction_errors.append(avg_pred_error)

        print(f"  Loss: {avg_loss:.6f}, Prediction Error: {avg_pred_error:.6f}")
        print(f"  Buffer size: {len(self.experience_buffer)}")

        training_log.append({
          'episode': episode + 1,
          'segment': segment_files[segment_idx].name,
          'loss': avg_loss,
          'prediction_error': avg_pred_error,
          'buffer_size': len(self.experience_buffer)
        })

        # Save model periodically
        if (episode + 1) % save_interval == 0:
          self.network.save(self.model_save_path)
          print(f"  Model saved to {self.model_save_path}")

          # Save training log
          log_path = Path(self.model_save_path).parent / "deep_learning_training_log.json"
          with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

      except Exception as e:
        print(f"  Error in episode {episode + 1}: {e}")
        import traceback
        traceback.print_exc()
        continue

      print()

    # Final save
    self.network.save(self.model_save_path)
    print(f"Training complete! Final model saved to {self.model_save_path}")

    # Save final training log
    log_path = Path(self.model_save_path).parent / "deep_learning_training_log.json"
    with open(log_path, 'w') as f:
      json.dump(training_log, f, indent=2)

    # Print summary
    if len(self.episode_losses) > 0:
      print("\nTraining Summary:")
      print(f"  Final avg loss: {np.mean(self.episode_losses[-10:]):.6f}")
      print(f"  Final avg prediction error: {np.mean(self.prediction_errors[-10:]):.6f}")
      print(f"  Best loss: {np.min(self.episode_losses):.6f}")
      print(f"  Total experiences collected: {len(self.experience_buffer)}")


def main():
  """Main training function."""
  # Create network
  network = DeepLearningControlNetwork(
    history_length=20,
    future_length=10,
    num_channels=7,
    conv_filters=(16, 32, 64),
    kernel_size=3,
    fc_dims=(128, 64),
    learning_rate=0.0001
  )

  # Create trainer
  trainer = DeepLearningControlTrainer(
    network=network,
    data_path="data",
    model_save_path="models/deep_learning_control.pkl",
    batch_size=32,
    buffer_capacity=5000
  )

  # Train
  print("Starting unsupervised deep learning control training...")
  print("=" * 60)
  trainer.train(num_episodes=50, save_interval=5)


if __name__ == "__main__":
  main()
