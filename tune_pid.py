import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tinyphysics import get_available_controllers, run_rollout
from tinyphysics_core.config import DATASET_PATH, DEFAULT_MODEL_PATH, ACC_G


def _load_segments(data_path: Path, num_segments: int) -> List[Path]:
  files = sorted(data_path.glob("*.csv"))
  if not files:
    raise FileNotFoundError(f"No CSV files found under {data_path}")
  return files[:min(num_segments, len(files))]


def _extract_segment_features(segment_path: Path) -> Dict[str, float]:
  """Extract statistical features from the raw data file."""
  df = pd.read_csv(segment_path)

  # Compute features from raw data
  features = {
    # Velocity statistics
    'avg_v_ego': np.mean(df['vEgo'].values),
    'std_v_ego': np.std(df['vEgo'].values),
    'max_v_ego': np.max(df['vEgo'].values),
    'min_v_ego': np.min(df['vEgo'].values),

    # Acceleration statistics
    'avg_a_ego': np.mean(df['aEgo'].values),
    'std_a_ego': np.std(df['aEgo'].values),
    'avg_abs_a_ego': np.mean(np.abs(df['aEgo'].values)),

    # Roll/lateral acceleration statistics
    'avg_roll': np.mean(df['roll'].values),
    'std_roll': np.std(df['roll'].values),
    'avg_abs_roll': np.mean(np.abs(df['roll'].values)),
    'avg_roll_lataccel': np.mean(np.sin(df['roll'].values) * ACC_G),
    'avg_abs_roll_lataccel': np.mean(np.abs(np.sin(df['roll'].values) * ACC_G)),

    # Target lateral acceleration statistics
    'avg_target_lataccel': np.mean(df['targetLateralAcceleration'].values),
    'std_target_lataccel': np.std(df['targetLateralAcceleration'].values),
    'avg_abs_target_lataccel': np.mean(np.abs(df['targetLateralAcceleration'].values)),
    'max_abs_target_lataccel': np.max(np.abs(df['targetLateralAcceleration'].values)),

    # Target lateral acceleration change rate (jerkiness of target)
    'avg_abs_target_lataccel_change': np.mean(np.abs(np.diff(df['targetLateralAcceleration'].values))),

    # Steer command statistics
    'avg_abs_steer_command': np.mean(np.abs(df['steerCommand'].values)),
    'std_steer_command': np.std(df['steerCommand'].values),
  }

  return features


def _collect_costs_with_features(segment_paths: List[Path], controller: str, model_path: Path) -> List[Dict[str, float]]:
  results: List[Dict[str, float]] = []
  for seg_path in segment_paths:
    rollout = run_rollout(seg_path, controller_type=controller, model_path=str(model_path), debug=False)

    # Extract features from the data file
    features = _extract_segment_features(seg_path)

    # Combine costs with features
    entry = {
      'segment': seg_path.stem,
      'lataccel_cost': rollout.cost['lataccel_cost'],
      'jerk_cost': rollout.cost['jerk_cost'],
      'total_cost': rollout.cost['total_cost'],
      **features,  # Add all extracted features
      **rollout.state_stats  # Add state statistics from rollout
    }
    results.append(entry)
  return results


def _collect_costs(segment_paths: List[Path], controller: str, model_path: Path) -> List[Dict[str, float]]:
  """Legacy function for backward compatibility."""
  results: List[Dict[str, float]] = []
  for seg_path in segment_paths:
    rollout = run_rollout(seg_path, controller_type=controller, model_path=str(model_path), debug=False)
    entry = {
      'segment': seg_path.stem,
      'lataccel_cost': rollout.cost['lataccel_cost'],
      'jerk_cost': rollout.cost['jerk_cost'],
      'total_cost': rollout.cost['total_cost']
    }
    results.append(entry)
  return results


def _init_centroids(data: np.ndarray, k: int, rng: random.Random) -> np.ndarray:
  indices = rng.sample(range(data.shape[0]), k)
  return data[indices].copy()


def _kmeans(data: np.ndarray, k: int, rng: random.Random, max_iters: int = 100, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
  if data.shape[0] < k:
    raise ValueError(f"Not enough samples ({data.shape[0]}) to form {k} clusters.")
  centroids = _init_centroids(data, k, rng)
  labels = np.zeros(data.shape[0], dtype=int)
  for _ in range(max_iters):
    distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
    new_labels = np.argmin(distances, axis=1)
    new_centroids = np.zeros_like(centroids)
    for cluster in range(k):
      members = data[new_labels == cluster]
      if members.size == 0:
        new_centroids[cluster] = data[rng.randrange(data.shape[0])]
      else:
        new_centroids[cluster] = members.mean(axis=0)
    shift = np.linalg.norm(new_centroids - centroids)
    centroids = new_centroids
    labels = new_labels
    if shift < tol:
      break
  return labels, centroids


def _analyze_feature_correlations(results: List[Dict[str, float]], controller_name: str) -> None:
  """Analyze which features correlate with controller performance."""
  print(f"\n=== Feature Correlation Analysis for {controller_name} ===")

  # Convert to DataFrame for easier analysis
  df = pd.DataFrame(results)

  # Features to analyze (exclude segment and cost columns)
  feature_cols = [col for col in df.columns if col not in ['segment', 'lataccel_cost', 'jerk_cost', 'total_cost']]

  if not feature_cols:
    print("No features available for correlation analysis")
    return

  # Compute correlations with total_cost
  correlations = {}
  for feature in feature_cols:
    corr = df[feature].corr(df['total_cost'])
    if not np.isnan(corr):
      correlations[feature] = corr

  # Sort by absolute correlation
  sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

  print(f"\nTop features correlated with total_cost (sorted by absolute correlation):")
  for feature, corr in sorted_corrs[:10]:
    print(f"  {feature:40s}: {corr:>7.4f}")

  # Find segments with highest and lowest costs
  best_segment = df.loc[df['total_cost'].idxmin()]
  worst_segment = df.loc[df['total_cost'].idxmax()]

  print(f"\nBest performing segment: {best_segment['segment']} (total_cost: {best_segment['total_cost']:.3f})")
  print(f"Worst performing segment: {worst_segment['segment']} (total_cost: {worst_segment['total_cost']:.3f})")

  # Compare feature values
  print(f"\nKey feature differences (best vs worst):")
  for feature in sorted_corrs[:5]:
    feature_name = feature[0]
    best_val = best_segment[feature_name]
    worst_val = worst_segment[feature_name]
    diff_pct = ((worst_val - best_val) / (abs(best_val) + 1e-9)) * 100
    print(f"  {feature_name:40s}: {best_val:>8.3f} vs {worst_val:>8.3f} ({diff_pct:>+7.1f}%)")


def _compare_controllers(results_dict: Dict[str, List[Dict[str, float]]]) -> None:
  """Compare performance between different controllers."""
  print("\n" + "="*80)
  print("=== Controller Comparison ===")
  print("="*80)

  # Summary statistics for each controller
  for controller_name, results in results_dict.items():
    df = pd.DataFrame(results)
    print(f"\n{controller_name}:")
    print(f"  Average total_cost: {df['total_cost'].mean():.4f} ± {df['total_cost'].std():.4f}")
    print(f"  Average lataccel_cost: {df['lataccel_cost'].mean():.4f} ± {df['lataccel_cost'].std():.4f}")
    print(f"  Average jerk_cost: {df['jerk_cost'].mean():.4f} ± {df['jerk_cost'].std():.4f}")
    print(f"  Best total_cost: {df['total_cost'].min():.4f}")
    print(f"  Worst total_cost: {df['total_cost'].max():.4f}")

  # If we have exactly 2 controllers, do detailed comparison
  if len(results_dict) == 2:
    controllers = list(results_dict.keys())
    c1_name, c2_name = controllers[0], controllers[1]
    df1 = pd.DataFrame(results_dict[c1_name])
    df2 = pd.DataFrame(results_dict[c2_name])

    # Ensure they have the same segments
    common_segments = set(df1['segment']).intersection(set(df2['segment']))
    if len(common_segments) > 0:
      df1_filtered = df1[df1['segment'].isin(common_segments)].sort_values('segment').reset_index(drop=True)
      df2_filtered = df2[df2['segment'].isin(common_segments)].sort_values('segment').reset_index(drop=True)

      cost_diff = df2_filtered['total_cost'] - df1_filtered['total_cost']
      c1_wins = (cost_diff > 0).sum()
      c2_wins = (cost_diff < 0).sum()
      ties = (cost_diff == 0).sum()

      print(f"\n{c1_name} vs {c2_name} (on {len(common_segments)} common segments):")
      print(f"  {c1_name} performs better: {c1_wins} segments ({100*c1_wins/len(common_segments):.1f}%)")
      print(f"  {c2_name} performs better: {c2_wins} segments ({100*c2_wins/len(common_segments):.1f}%)")
      print(f"  Ties: {ties} segments")

      # Identify where each controller excels
      feature_cols = [col for col in df1.columns if col not in ['segment', 'lataccel_cost', 'jerk_cost', 'total_cost']]

      if feature_cols:
        # Find segments where controller 1 is significantly better
        c1_better_idx = cost_diff > cost_diff.median()
        c2_better_idx = cost_diff < -cost_diff.median()

        print(f"\nScenarios where {c1_name} excels:")
        for feature in feature_cols[:5]:
          c1_better_mean = df1_filtered.loc[c1_better_idx, feature].mean()
          c2_better_mean = df1_filtered.loc[c2_better_idx, feature].mean()
          if not np.isnan(c1_better_mean) and not np.isnan(c2_better_mean):
            diff = c1_better_mean - c2_better_mean
            print(f"  {feature:40s}: {c1_better_mean:8.3f} (vs {c2_better_mean:8.3f} when {c2_name} better)")


def summarize_clusters(results: List[Dict[str, float]], labels: np.ndarray, centroids: np.ndarray, show_features: bool = False) -> None:
  print("\n=== Cluster Summary ===")
  feature_names = ['lataccel_cost', 'jerk_cost', 'total_cost']

  # Get all available features if requested
  if show_features and len(results) > 0:
    all_feature_names = [k for k in results[0].keys() if k not in ['segment', 'lataccel_cost', 'jerk_cost', 'total_cost']]
  else:
    all_feature_names = []

  for cluster_id in range(centroids.shape[0]):
    cluster_entries = [entry for entry, label in zip(results, labels) if label == cluster_id]
    count = len(cluster_entries)
    if count == 0:
      print(f"Cluster {cluster_id}: empty")
      continue
    print(f"\nCluster {cluster_id}: {count} segments")
    centroid = centroids[cluster_id]
    print("  Centroid:", ", ".join(f"{name}={value:.3f}" for name, value in zip(feature_names, centroid)))
    avg_total = np.mean([entry['total_cost'] for entry in cluster_entries])
    print(f"  Average total_cost: {avg_total:.3f}")

    # Show characteristic features of this cluster
    if show_features and all_feature_names:
      cluster_df = pd.DataFrame(cluster_entries)
      print("  Cluster characteristics:")
      for feature in all_feature_names[:5]:  # Show top 5 features
        avg_val = cluster_df[feature].mean()
        print(f"    {feature:40s}: {avg_val:8.3f}")

    top_segments = sorted(cluster_entries, key=lambda e: e['total_cost'])[:3]
    print("  Representative segments:")
    for entry in top_segments:
      print(f"    segment {entry['segment']}: lataccel={entry['lataccel_cost']:.3f}, jerk={entry['jerk_cost']:.3f}, total={entry['total_cost']:.3f}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Run PID controller across segments and cluster by cost behavior.")
  parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
  parser.add_argument("--data_path", type=Path, default=DATASET_PATH)
  parser.add_argument("--controller", default="pid", choices=get_available_controllers())
  parser.add_argument("--num_segments", type=int, default=100)
  parser.add_argument("--clusters", type=int, default=3)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--analyze", action='store_true', help="Perform deep feature analysis")
  parser.add_argument("--compare", action='store_true', help="Compare PID vs PID_w_FF controllers")
  parser.add_argument("--controllers", nargs='+', choices=get_available_controllers(), help="Specific controllers to compare")
  args = parser.parse_args()

  rng = random.Random(args.seed)
  segment_paths = _load_segments(args.data_path, args.num_segments)

  if args.compare:
    # Compare both controllers (or more if specified)
    if hasattr(args, 'controllers') and args.controllers:
      controllers_to_compare = args.controllers
    else:
      controllers_to_compare = ['pid', 'pid_w_ff']
    results_dict = {}

    for controller in controllers_to_compare:
      print(f"\nRunning controller '{controller}' on {len(segment_paths)} segments...")
      if args.analyze:
        results = _collect_costs_with_features(segment_paths, controller, args.model_path)
      else:
        results = _collect_costs(segment_paths, controller, args.model_path)
      results_dict[controller] = results

    # Perform comparison analysis
    _compare_controllers(results_dict)

    # Analyze features for each controller
    if args.analyze:
      for controller, results in results_dict.items():
        _analyze_feature_correlations(results, controller)

      # Also do clustering for each
      for controller, results in results_dict.items():
        print(f"\n{'='*80}")
        print(f"Clustering analysis for {controller}")
        print(f"{'='*80}")
        data = np.array([[entry['lataccel_cost'], entry['jerk_cost'], entry['total_cost']] for entry in results])
        labels, centroids = _kmeans(data, args.clusters, rng)
        summarize_clusters(results, labels, centroids, show_features=True)

  else:
    # Single controller analysis
    print(f"Running controller '{args.controller}' on {len(segment_paths)} segments...")
    if args.analyze:
      results = _collect_costs_with_features(segment_paths, args.controller, args.model_path)
      _analyze_feature_correlations(results, args.controller)
    else:
      results = _collect_costs(segment_paths, args.controller, args.model_path)

    data = np.array([[entry['lataccel_cost'], entry['jerk_cost'], entry['total_cost']] for entry in results])
    labels, centroids = _kmeans(data, args.clusters, rng)
    summarize_clusters(results, labels, centroids, show_features=args.analyze)


if __name__ == "__main__":
  main()
