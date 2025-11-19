"""
Self-supervised training for adaptive PID gain network.

This script implements a self-supervised learning approach where:
1. The network explores different gain combinations
2. Experiences are collected in a replay buffer with associated costs
3. The network learns to predict gains that minimize cost through imitation of best experiences
"""
import argparse
import importlib
import numpy as np
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import json

from tqdm import tqdm

from gain_adapter_network import GainAdapterNetwork, ReplayBuffer
from controllers.pid_adaptive import Controller, AdaptiveGains
from tinyphysics import run_rollout
from tinyphysics_core.config import DATASET_PATH, DEFAULT_MODEL_PATH


@dataclass
class TrainingConfig:
  """Configuration for training."""
  num_epochs: int = 50
  episodes_per_epoch: int = 20
  batch_size: int = 32
  learning_rate: float = 0.001
  buffer_capacity: int = 10000
  exploration_rate: float = 0.3
  exploration_decay: float = 0.95
  best_sample_ratio: float = 0.7  # Sample 70% from best experiences
  network_path: str = "models/gain_adapter.pkl"
  segment_limit: int = 100  # Limit segments for faster training


class AdaptiveGainTrainer:
  """Trainer for adaptive gain network using self-supervised learning."""

  def __init__(self, config: TrainingConfig, data_paths: List[Path]):
    self.config = config
    self.data_paths = data_paths[:config.segment_limit]
    self.network = GainAdapterNetwork(learning_rate=config.learning_rate)
    self.buffer = ReplayBuffer(capacity=config.buffer_capacity)
    self.exploration_rate = config.exploration_rate

    # Gain exploration ranges
    self.gain_ranges = {
      'p': (0.1, 0.8),
      'i': (0.0, 0.2),
      'd': (-0.3, 0.0),
      'feedforward_gain': (0.3, 1.2),
    }

  def _random_gains(self) -> AdaptiveGains:
    """Generate random gains for exploration."""
    return AdaptiveGains(
      p=np.random.uniform(*self.gain_ranges['p']),
      i=np.random.uniform(*self.gain_ranges['i']),
      d=np.random.uniform(*self.gain_ranges['d']),
      feedforward_gain=np.random.uniform(*self.gain_ranges['feedforward_gain'])
    )

  def _explore_or_exploit(self, features: np.ndarray) -> AdaptiveGains:
    """Choose between exploration (random gains) or exploitation (network prediction)."""
    if np.random.random() < self.exploration_rate:
      return self._random_gains()
    else:
      return self.network.predict_gains(features)

  def collect_experience(self, segment_path: Path, use_exploration: bool = True) -> Tuple[List, float]:
    """
    Run a single segment and collect experience.

    Returns:
      experiences: List of (features, gains, step_cost) tuples
      total_cost: Total cost for the segment
    """
    # Create controller with custom gain adapter
    controller = Controller()
    controller.use_adaptive_gains = True

    experiences = []
    step_costs = []

    # Custom gain adapter that logs experiences
    class ExperienceCollector:
      def __init__(self, trainer, use_exploration):
        self.trainer = trainer
        self.use_exploration = use_exploration
        self.features_log = []
        self.gains_log = []

      def predict_gains(self, features):
        if self.use_exploration:
          gains = self.trainer._explore_or_exploit(features)
        else:
          gains = self.trainer.network.predict_gains(features)

        self.features_log.append(features.copy())
        self.gains_log.append((gains.p, gains.i, gains.d, gains.feedforward_gain))
        return gains

    collector = ExperienceCollector(self, use_exploration)
    controller.gain_adapter = collector

    # Run simulation
    try:
      result = run_rollout(segment_path, 'pid_adaptive', DEFAULT_MODEL_PATH, debug=False)
      total_cost = result.cost['total_cost']

      # Extract per-step costs from cost_history
      if 'total_cost' in result.cost_history:
        step_costs = result.cost_history['total_cost']

      # Create experiences with per-step costs
      for i, (features, gains) in enumerate(zip(collector.features_log, collector.gains_log)):
        step_cost = step_costs[i] if i < len(step_costs) else total_cost / max(1, len(collector.features_log))
        experiences.append((features, np.array(gains, dtype=np.float32), step_cost))

      return experiences, total_cost

    except Exception as e:
      print(f"Error running segment {segment_path}: {e}")
      return [], float('inf')

  def train_epoch(self, epoch: int) -> dict:
    """Train for one epoch."""
    epoch_stats = {
      'avg_cost': 0.0,
      'min_cost': float('inf'),
      'max_cost': 0.0,
      'avg_loss': 0.0,
      'exploration_rate': self.exploration_rate,
    }

    # Collect experiences
    print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
    print("Collecting experiences...")

    costs = []
    for i, segment_path in enumerate(tqdm(self.data_paths[:self.config.episodes_per_epoch], desc="Rollouts")):
      experiences, total_cost = self.collect_experience(segment_path, use_exploration=True)

      # Add experiences to buffer
      for exp in experiences:
        self.buffer.push(exp)

      costs.append(total_cost)

    # Update statistics
    epoch_stats['avg_cost'] = np.mean(costs)
    epoch_stats['min_cost'] = np.min(costs)
    epoch_stats['max_cost'] = np.max(costs)

    # Train on collected experiences
    if len(self.buffer) >= self.config.batch_size:
      print("Training network...")
      losses = []

      # Multiple training iterations per epoch
      num_train_steps = min(len(self.buffer) // self.config.batch_size, 50)

      for _ in tqdm(range(num_train_steps), desc="Training"):
        # Sample from buffer (prioritize best experiences)
        n_best = int(self.config.batch_size * self.config.best_sample_ratio)
        best_batch = self.buffer.get_best_experiences(n_best)
        random_batch = self.buffer.sample(self.config.batch_size - n_best)
        batch = best_batch + random_batch

        # Train on batch
        for features, gains, cost in batch:
          # Normalize features
          features_norm = features / (np.abs(features).max() + 1e-8)

          # Normalize gains to [-1, 1] range
          target_normalized = (gains - self.network.gain_means) / (self.network.gain_stds + 1e-8)
          target_normalized = np.clip(target_normalized, -1, 1)

          # Forward and backward pass
          _, activations = self.network.forward(features_norm)
          loss = self.network.backward(activations, target_normalized)
          losses.append(loss)

      epoch_stats['avg_loss'] = np.mean(losses)

      # Update gain statistics based on best experiences
      best_gains = [exp[1] for exp in self.buffer.get_best_experiences(100)]
      if len(best_gains) > 0:
        self.network.update_gain_statistics(best_gains)

    # Decay exploration rate
    self.exploration_rate *= self.config.exploration_decay
    self.exploration_rate = max(0.05, self.exploration_rate)  # Minimum 5% exploration

    return epoch_stats

  def evaluate(self, num_segments: int = 10) -> dict:
    """Evaluate current network on test segments."""
    print("\nEvaluating network...")
    test_paths = self.data_paths[-num_segments:]

    costs_with_network = []
    costs_without_network = []

    for segment_path in tqdm(test_paths, desc="Evaluation"):
      # Test with network
      experiences, cost_with = self.collect_experience(segment_path, use_exploration=False)
      costs_with_network.append(cost_with)

      # Test without network (baseline gains)
      controller = Controller()
      controller.use_adaptive_gains = False
      try:
        result = run_rollout(segment_path, 'pid_adaptive', DEFAULT_MODEL_PATH, debug=False)
        costs_without_network.append(result.cost['total_cost'])
      except:
        costs_without_network.append(float('inf'))

    eval_stats = {
      'avg_cost_with_network': np.mean(costs_with_network),
      'avg_cost_baseline': np.mean(costs_without_network),
      'improvement': np.mean(costs_without_network) - np.mean(costs_with_network),
      'improvement_pct': 100 * (1 - np.mean(costs_with_network) / np.mean(costs_without_network)),
    }

    return eval_stats

  def train(self) -> None:
    """Main training loop."""
    print(f"Starting training with {len(self.data_paths)} segments")
    print(f"Config: {self.config}")

    training_history = []

    for epoch in range(self.config.num_epochs):
      epoch_stats = self.train_epoch(epoch)

      # Evaluate every 5 epochs
      if (epoch + 1) % 5 == 0:
        eval_stats = self.evaluate()
        epoch_stats.update(eval_stats)
        print(f"\nEvaluation Results:")
        print(f"  With Network: {eval_stats['avg_cost_with_network']:.4f}")
        print(f"  Baseline: {eval_stats['avg_cost_baseline']:.4f}")
        print(f"  Improvement: {eval_stats['improvement_pct']:.2f}%")

      training_history.append(epoch_stats)

      # Print epoch summary
      print(f"\nEpoch {epoch + 1} Summary:")
      print(f"  Avg Cost: {epoch_stats['avg_cost']:.4f}")
      print(f"  Min Cost: {epoch_stats['min_cost']:.4f}")
      print(f"  Avg Loss: {epoch_stats.get('avg_loss', 0.0):.6f}")
      print(f"  Exploration Rate: {epoch_stats['exploration_rate']:.3f}")
      print(f"  Buffer Size: {len(self.buffer)}")

      # Save checkpoint
      if (epoch + 1) % 10 == 0:
        checkpoint_path = f"models/gain_adapter_epoch_{epoch + 1}.pkl"
        Path(checkpoint_path).parent.mkdir(exist_ok=True)
        self.network.save(checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")

    # Save final model
    Path(self.config.network_path).parent.mkdir(exist_ok=True)
    self.network.save(self.config.network_path)
    print(f"\nTraining complete! Saved model to {self.config.network_path}")

    # Save training history
    history_path = Path(self.config.network_path).with_suffix('.json')
    with open(history_path, 'w') as f:
      json.dump(training_history, f, indent=2)
    print(f"Saved training history to {history_path}")


def main():
  parser = argparse.ArgumentParser(description="Train adaptive PID gain network")
  parser.add_argument("--data_path", type=str, default=str(DATASET_PATH), help="Path to dataset directory")
  parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
  parser.add_argument("--episodes_per_epoch", type=int, default=20, help="Segments to run per epoch")
  parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
  parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
  parser.add_argument("--segment_limit", type=int, default=100, help="Maximum segments to use")
  parser.add_argument("--output_path", type=str, default="models/gain_adapter.pkl", help="Output model path")

  args = parser.parse_args()

  # Get data paths
  data_path = Path(args.data_path)
  if data_path.is_dir():
    data_paths = sorted(list(data_path.glob("*.csv")))
  else:
    raise ValueError(f"Data path {data_path} is not a directory")

  if len(data_paths) == 0:
    raise ValueError(f"No CSV files found in {data_path}")

  print(f"Found {len(data_paths)} segments")

  # Create config
  config = TrainingConfig(
    num_epochs=args.num_epochs,
    episodes_per_epoch=args.episodes_per_epoch,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    segment_limit=args.segment_limit,
    network_path=args.output_path,
  )

  # Train
  trainer = AdaptiveGainTrainer(config, data_paths)
  trainer.train()


if __name__ == "__main__":
  main()
