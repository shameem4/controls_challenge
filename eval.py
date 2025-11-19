import argparse
from functools import partial
from pathlib import Path
from typing import Dict, List

import pandas as pd
import seaborn as sns
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tinyphysics import get_available_controllers, run_rollout
from tinyphysics_core.config import CONTROL_START_IDX, DATASET_PATH, DEFAULT_MODEL_PATH, DEL_T, LAT_ACCEL_COST_MULTIPLIER
from tinyphysics_core.report import DEFAULT_COLORS, build_report_html, save_report, save_metrics_csv

sns.set_theme()
SAMPLE_ROLLOUTS = 5

COLORS = DEFAULT_COLORS


def _build_summary_row(segment: str, controller_label: str, result) -> Dict[str, float]:
  row: Dict[str, float] = {
    'row_type': 'summary',
    'segment': segment,
    'controller': controller_label
  }
  row.update(result.cost)
  row.update(result.diagnostics)
  row.update(result.state_stats)
  return row


def _build_step_rows(segment: str, controller_label: str, result) -> List[Dict[str, float]]:
  rows: List[Dict[str, float]] = []
  target = result.target_lataccel_history
  actual = result.current_lataccel_history
  actions = result.action_history
  for idx in range(CONTROL_START_IDX, len(actual)):
    prev_lataccel = actual[idx - 1] if idx > 0 else actual[idx]
    jerk = (actual[idx] - prev_lataccel) / DEL_T
    lataccel_loss = ((target[idx] - actual[idx]) ** 2) * 100
    jerk_loss = (jerk ** 2) * 100
    total_loss = (lataccel_loss * LAT_ACCEL_COST_MULTIPLIER) + jerk_loss
    rows.append({
      'row_type': 'step',
      'segment': segment,
      'controller': controller_label,
      'step': idx,
      'desired_lataccel': target[idx],
      'controller_lataccel': actual[idx],
      'lat_error': target[idx] - actual[idx],
      'action': actions[idx] if idx < len(actions) else 0.0,
      'jerk': jerk,
      'lataccel_cost': lataccel_loss,
      'jerk_cost': jerk_loss,
      'total_cost': total_loss,
      'step_lataccel_cost': lataccel_loss,
      'step_jerk_cost': jerk_loss,
      'step_total_cost': total_loss
    })
  return rows


def _record_result(rows: List[Dict[str, float]], step_rows: List[Dict[str, float]], segment: str, controller_label: str, result) -> None:
  rows.append(_build_summary_row(segment, controller_label, result))
  step_rows.extend(_build_step_rows(segment, controller_label, result))


if __name__ == "__main__":
  available_controllers = get_available_controllers()
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
  parser.add_argument("--data_path", type=str, default=str(DATASET_PATH))
  parser.add_argument("--num_segs", type=int, default=100)
  parser.add_argument("--test_controller", default='pid_opt', choices=available_controllers)
  parser.add_argument("--baseline_controller", default='pid_w_ff', choices=available_controllers)
  args = parser.parse_args()

  data_path = Path(args.data_path)
  assert data_path.is_dir(), "data_path should be a directory"

  summary_rows: List[Dict[str, float]] = []
  step_rows: List[Dict[str, float]] = []
  sample_rollouts = []
  files = sorted(data_path.iterdir())[:args.num_segs]
  print("Running rollouts for visualizations...")
  for d, data_file in enumerate(tqdm(files[:SAMPLE_ROLLOUTS], total=SAMPLE_ROLLOUTS)):
    test_result = run_rollout(data_file, args.test_controller, args.model_path, debug=False)
    baseline_result = run_rollout(data_file, args.baseline_controller, args.model_path, debug=False)
    sample_rollouts.append({
      'seg': data_file.stem,
      'test_controller': args.test_controller,
      'baseline_controller': args.baseline_controller,
      'desired_lataccel': test_result.target_lataccel_history,
      'test_controller_lataccel': test_result.current_lataccel_history,
      'baseline_controller_lataccel': baseline_result.current_lataccel_history,
    })

    _record_result(summary_rows, step_rows, data_file.stem, 'test', test_result)
    _record_result(summary_rows, step_rows, data_file.stem, 'baseline', baseline_result)

  for controller_cat, controller_type in [('baseline', args.baseline_controller), ('test', args.test_controller)]:
    print(f"Running batch rollouts => {controller_cat} controller: {controller_type}")
    rollout_partial = partial(run_rollout, controller_type=controller_type, model_path=args.model_path, debug=False)
    segment_files = files[SAMPLE_ROLLOUTS:]
    results = process_map(rollout_partial, segment_files, max_workers=16, chunksize=10)
    for seg, result in zip(segment_files, results):
      _record_result(summary_rows, step_rows, seg.stem, controller_cat, result)

  combined_rows = summary_rows + step_rows
  report_html = build_report_html(
    args.test_controller,
    args.baseline_controller,
    sample_rollouts,
    summary_rows,
    len(files),
    colors=COLORS,
    expected_sample_plots=SAMPLE_ROLLOUTS
  )
  save_report(report_html)
  save_metrics_csv(combined_rows)
  costs_df = pd.DataFrame(combined_rows)
  summary_df = costs_df[costs_df['row_type'] == 'summary']
  print("\nCost Summary by controller:")
  print(summary_df.groupby('controller')['total_cost'].describe())
  for controller in summary_df['controller'].unique():
    top_segments = summary_df[summary_df['controller'] == controller].nlargest(3, 'total_cost')
    print(f"\nTop segments for {controller}:")
    print(top_segments[['segment', 'total_cost', 'lataccel_cost', 'jerk_cost']])
  print("Report saved to: './report.html'")
