import argparse
import seaborn as sns


from functools import partial
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tinyphysics import get_available_controllers, run_rollout
from tinyphysics_core.config import DATASET_PATH, DEFAULT_MODEL_PATH
from tinyphysics_core.report import DEFAULT_COLORS, build_report_html, save_report, save_costs_csv, save_diagnostics_csv

sns.set_theme()
SAMPLE_ROLLOUTS = 5

COLORS = DEFAULT_COLORS


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

  costs = []
  diagnostics = []
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

    costs.append({'segment': data_file.stem, 'controller': 'test', **test_result.cost})
    costs.append({'segment': data_file.stem, 'controller': 'baseline', **baseline_result.cost})
    diagnostics.append({'segment': data_file.stem, 'controller': 'test', **test_result.diagnostics})
    diagnostics.append({'segment': data_file.stem, 'controller': 'baseline', **baseline_result.diagnostics})

  for controller_cat, controller_type in [('baseline', args.baseline_controller), ('test', args.test_controller)]:
    print(f"Running batch rollouts => {controller_cat} controller: {controller_type}")
    rollout_partial = partial(run_rollout, controller_type=controller_type, model_path=args.model_path, debug=False)
    segment_files = files[SAMPLE_ROLLOUTS:]
    results = process_map(rollout_partial, segment_files, max_workers=16, chunksize=10)
    for seg, result in zip(segment_files, results):
      costs.append({'segment': seg.stem, 'controller': controller_cat, **result.cost})
      diagnostics.append({'segment': seg.stem, 'controller': controller_cat, **result.diagnostics})

  report_html = build_report_html(
    args.test_controller,
    args.baseline_controller,
    sample_rollouts,
    costs,
    len(files),
    colors=COLORS,
    expected_sample_plots=SAMPLE_ROLLOUTS
  )
  save_report(report_html)
  save_costs_csv(costs)
  save_diagnostics_csv(diagnostics)
  print("Report saved to: './report.html'")
