"""
Evaluate adaptive PID controller with trained gain network.
"""
import argparse
from pathlib import Path
import numpy as np
from typing import List
import json

from gain_adapter_network import GainAdapterNetwork
from controllers.pid_adaptive import Controller
from tinyphysics import run_rollout
from tinyphysics_core.config import DATASET_PATH, DEFAULT_MODEL_PATH


def evaluate_with_network(data_paths: List[Path], network_path: str, baseline: bool = False) -> dict:
  """
  Evaluate controller performance.

  Args:
    data_paths: List of segment paths to evaluate
    network_path: Path to trained network
    baseline: If True, use baseline gains instead of network

  Returns:
    Dictionary with evaluation results
  """
  # Load network
  network = None
  if not baseline and Path(network_path).exists():
    network = GainAdapterNetwork.load(network_path)
    print(f"Loaded network from {network_path}")
  elif not baseline:
    print(f"Warning: Network file {network_path} not found, using baseline gains")
    baseline = True

  # Monkey-patch the controller module to use our network
  if network is not None:
    import controllers.pid_adaptive as pid_adaptive_module
    # Store original Controller class
    original_controller = pid_adaptive_module.Controller

    # Create a wrapper that injects our network
    class NetworkController(original_controller):
      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_adaptive_gains = True
        self.gain_adapter = network

    # Replace the Controller class
    pid_adaptive_module.Controller = NetworkController

  results = []
  costs = []

  print(f"\nEvaluating {'baseline' if baseline else 'adaptive'} controller on {len(data_paths)} segments...")

  for segment_path in data_paths:
    try:
      result = run_rollout(segment_path, 'pid_adaptive', DEFAULT_MODEL_PATH, debug=False)

      segment_result = {
        'segment': segment_path.stem,
        'lataccel_cost': result.cost['lataccel_cost'],
        'jerk_cost': result.cost['jerk_cost'],
        'total_cost': result.cost['total_cost'],
      }

      # Add diagnostics if available
      if hasattr(result, 'diagnostics') and result.diagnostics:
        segment_result['diagnostics'] = result.diagnostics

      results.append(segment_result)
      costs.append(result.cost['total_cost'])

      print(f"  {segment_path.stem}: total={result.cost['total_cost']:.4f}, "
            f"lataccel={result.cost['lataccel_cost']:.4f}, jerk={result.cost['jerk_cost']:.4f}")

    except Exception as e:
      print(f"  Error on {segment_path.stem}: {e}")
      costs.append(float('inf'))

  # Compute summary statistics
  valid_costs = [c for c in costs if c != float('inf')]

  summary = {
    'num_segments': len(data_paths),
    'successful_segments': len(valid_costs),
    'avg_total_cost': np.mean(valid_costs) if valid_costs else float('inf'),
    'std_total_cost': np.std(valid_costs) if valid_costs else 0.0,
    'min_total_cost': np.min(valid_costs) if valid_costs else float('inf'),
    'max_total_cost': np.max(valid_costs) if valid_costs else float('inf'),
    'avg_lataccel_cost': np.mean([r['lataccel_cost'] for r in results]),
    'avg_jerk_cost': np.mean([r['jerk_cost'] for r in results]),
  }

  return {
    'summary': summary,
    'segments': results,
  }


def compare_controllers(data_paths: List[Path], network_path: str) -> None:
  """Compare adaptive controller with baseline."""
  print("=" * 60)
  print("BASELINE CONTROLLER")
  print("=" * 60)
  baseline_results = evaluate_with_network(data_paths, network_path, baseline=True)

  print("\n" + "=" * 60)
  print("ADAPTIVE CONTROLLER")
  print("=" * 60)
  adaptive_results = evaluate_with_network(data_paths, network_path, baseline=False)

  # Print comparison
  print("\n" + "=" * 60)
  print("COMPARISON")
  print("=" * 60)

  baseline_cost = baseline_results['summary']['avg_total_cost']
  adaptive_cost = adaptive_results['summary']['avg_total_cost']
  improvement = baseline_cost - adaptive_cost
  improvement_pct = 100 * improvement / baseline_cost if baseline_cost > 0 else 0

  print(f"\nBaseline Avg Cost:  {baseline_cost:.4f}")
  print(f"Adaptive Avg Cost:  {adaptive_cost:.4f}")
  print(f"Improvement:        {improvement:.4f} ({improvement_pct:+.2f}%)")

  print(f"\nBaseline Lataccel:  {baseline_results['summary']['avg_lataccel_cost']:.4f}")
  print(f"Adaptive Lataccel:  {adaptive_results['summary']['avg_lataccel_cost']:.4f}")

  print(f"\nBaseline Jerk:      {baseline_results['summary']['avg_jerk_cost']:.4f}")
  print(f"Adaptive Jerk:      {adaptive_results['summary']['avg_jerk_cost']:.4f}")

  # Save detailed results
  output = {
    'baseline': baseline_results,
    'adaptive': adaptive_results,
    'comparison': {
      'improvement': improvement,
      'improvement_pct': improvement_pct,
    }
  }

  output_path = Path("results/adaptive_comparison.json")
  output_path.parent.mkdir(exist_ok=True)
  with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

  print(f"\nDetailed results saved to {output_path}")


def main():
  parser = argparse.ArgumentParser(description="Evaluate adaptive PID controller")
  parser.add_argument("--data_path", type=str, default=str(DATASET_PATH), help="Path to dataset directory")
  parser.add_argument("--network_path", type=str, default="models/gain_adapter.pkl", help="Path to trained network")
  parser.add_argument("--num_segments", type=int, default=None, help="Number of segments to evaluate (default: all)")
  parser.add_argument("--baseline", action="store_true", help="Evaluate baseline controller only")
  parser.add_argument("--compare", action="store_true", help="Compare baseline and adaptive controllers")
  parser.add_argument("--segments", type=str, nargs='+', help="Specific segment IDs to evaluate")

  args = parser.parse_args()

  # Get data paths
  data_path = Path(args.data_path)
  if data_path.is_dir():
    if args.segments:
      # Evaluate specific segments
      data_paths = [data_path / f"{seg}.csv" for seg in args.segments]
      data_paths = [p for p in data_paths if p.exists()]
    else:
      data_paths = sorted(list(data_path.glob("*.csv")))
  else:
    raise ValueError(f"Data path {data_path} is not a directory")

  if len(data_paths) == 0:
    raise ValueError(f"No CSV files found in {data_path}")

  # Limit segments if specified
  if args.num_segments is not None:
    data_paths = data_paths[:args.num_segments]

  print(f"Found {len(data_paths)} segments to evaluate")

  # Run evaluation
  if args.compare:
    compare_controllers(data_paths, args.network_path)
  else:
    results = evaluate_with_network(data_paths, args.network_path, baseline=args.baseline)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, value in results['summary'].items():
      print(f"{key:25s}: {value}")


if __name__ == "__main__":
  main()
