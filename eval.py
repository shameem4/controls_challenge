import argparse
import seaborn as sns


from functools import partial
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tinyphysics import get_available_controllers, run_rollout
from tinyphysics_core.config import DATASET_PATH, DEFAULT_MODEL_PATH
from tinyphysics_core.report import DEFAULT_COLORS, build_report_html, save_report

sns.set_theme()
SAMPLE_ROLLOUTS = 5

COLORS = DEFAULT_COLORS


if __name__ == "__main__":
  available_controllers = get_available_controllers()
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
  parser.add_argument("--data_path", type=str, default=str(DATASET_PATH))
  parser.add_argument("--num_segs", type=int, default=100)
  parser.add_argument("--test_controller", default='pid', choices=available_controllers)
  parser.add_argument("--baseline_controller", default='pid', choices=available_controllers)
  args = parser.parse_args()

  data_path = Path(args.data_path)
  assert data_path.is_dir(), "data_path should be a directory"

  costs = []
  sample_rollouts = []
  files = sorted(data_path.iterdir())[:args.num_segs]
  print("Running rollouts for visualizations...")
  for d, data_file in enumerate(tqdm(files[:SAMPLE_ROLLOUTS], total=SAMPLE_ROLLOUTS)):
    test_cost, test_target_lataccel, test_current_lataccel = run_rollout(data_file, args.test_controller, args.model_path, debug=False)
    baseline_cost, baseline_target_lataccel, baseline_current_lataccel = run_rollout(data_file, args.baseline_controller, args.model_path, debug=False)
    sample_rollouts.append({
      'seg': data_file.stem,
      'test_controller': args.test_controller,
      'baseline_controller': args.baseline_controller,
      'desired_lataccel': test_target_lataccel,
      'test_controller_lataccel': test_current_lataccel,
      'baseline_controller_lataccel': baseline_current_lataccel,
    })

    costs.append({'controller': 'test', **test_cost})
    costs.append({'controller': 'baseline', **baseline_cost})

  for controller_cat, controller_type in [('baseline', args.baseline_controller), ('test', args.test_controller)]:
    print(f"Running batch rollouts => {controller_cat} controller: {controller_type}")
    rollout_partial = partial(run_rollout, controller_type=controller_type, model_path=args.model_path, debug=False)
    results = process_map(rollout_partial, files[SAMPLE_ROLLOUTS:], max_workers=16, chunksize=10)
    costs += [{'controller': controller_cat, **result[0]} for result in results]

  report_html = build_report_html(args.test_controller, args.baseline_controller, sample_rollouts, costs, len(files), colors=COLORS)
  save_report(report_html)
  print("Report saved to: './report.html'")
