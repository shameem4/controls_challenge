"""
Quick start script for training adaptive gains on worst-performing segments.

This script focuses training on the problematic high-speed segments identified
in the cluster analysis.
"""
import argparse
from pathlib import Path
from train_adaptive_gains import AdaptiveGainTrainer, TrainingConfig
from tinyphysics_core.config import DATASET_PATH

# Worst performing segments from Cluster 0 (high speed, high jerk)
WORST_SEGMENTS = [
  "00000", "00006", "00012", "00014", "00019", "00021", "00026", "00027",
  "00030", "00032", "00036", "00046", "00052", "00054", "00056", "00058",
  "00063", "00069", "00071", "00077", "00082", "00089", "00091", "00093",
  "00094", "00095"
]


def main():
  parser = argparse.ArgumentParser(description="Quick train on worst segments")
  parser.add_argument("--data_path", type=str, default=str(DATASET_PATH), help="Dataset directory")
  parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
  parser.add_argument("--output", type=str, default="models/gain_adapter_targeted.pkl", help="Output path")
  parser.add_argument("--all_segments", action="store_true", help="Use all segments, not just worst")
  parser.add_argument("--num_segments", type=int, default=None, help="Number of segments to use (default: all worst segments or all available)")

  args = parser.parse_args()

  data_path = Path(args.data_path)

  if args.all_segments:
    # Use all available segments
    data_paths = sorted(list(data_path.glob("*.csv")))
    print(f"Found {len(data_paths)} total segments")
  else:
    # Focus on worst segments
    data_paths = [data_path / f"{seg}.csv" for seg in WORST_SEGMENTS]
    data_paths = [p for p in data_paths if p.exists()]
    print(f"Found {len(data_paths)} worst-performing segments")

  # Limit number of segments if specified
  if args.num_segments is not None:
    data_paths = data_paths[:args.num_segments]
    print(f"Using {len(data_paths)} segments for training")

  if not args.all_segments:
    print("Segments: " + ", ".join([p.stem for p in data_paths]))

  if len(data_paths) == 0:
    raise ValueError(f"No segments found in {data_path}")

  # Configuration optimized for targeted training
  config = TrainingConfig(
    num_epochs=args.num_epochs,
    episodes_per_epoch=len(data_paths),  # Use all segments each epoch
    batch_size=32,
    learning_rate=0.0005,  # Conservative learning rate to prevent instability
    buffer_capacity=5000,
    exploration_rate=0.4,  # Higher initial exploration
    exploration_decay=0.94,
    best_sample_ratio=0.8,  # Focus more on best experiences
    network_path=args.output,
    segment_limit=len(data_paths),
  )

  print(f"\nTraining configuration:")
  print(f"  Epochs: {config.num_epochs}")
  print(f"  Episodes per epoch: {config.episodes_per_epoch}")
  print(f"  Batch size: {config.batch_size}")
  print(f"  Learning rate: {config.learning_rate}")
  print(f"  Initial exploration: {config.exploration_rate}")
  print(f"  Output: {config.network_path}\n")

  # Train (always resume from best checkpoint in quick train)
  trainer = AdaptiveGainTrainer(config, data_paths, resume=True)
  trainer.train()

  # Also copy best checkpoint to output path if different
  best_checkpoint = Path(config.network_path).parent / "gain_adapter_best.pkl"
  if best_checkpoint.exists() and str(best_checkpoint) != config.network_path:
    import shutil
    shutil.copy(best_checkpoint, config.network_path)
    print(f"\nCopied best checkpoint to {config.network_path}")

  print("\n" + "=" * 60)
  print("Training complete!")
  print("=" * 60)
  print(f"\nModel saved to: {config.network_path}")
  print("\nTo evaluate, run:")
  print(f"  python eval_adaptive.py --network_path {config.network_path} --compare")
  print("\nOr test on specific segments:")
  print(f"  python eval_adaptive.py --network_path {config.network_path} --segments 00094 00071 00063")


if __name__ == "__main__":
  main()
