import argparse
import importlib
import os
import signal
import urllib.request
import zipfile
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.contrib.concurrent import process_map

from tinyphysics_core.config import (
  CONTROL_START_IDX,
  DATASET_PATH,
  DATASET_URL,
  DEFAULT_MODEL_PATH,
  HISTOGRAM_BINS,
  BASE_DIR,
)
from tinyphysics_core.model import TinyPhysicsModel
from tinyphysics_core.runner import RolloutRunner
from tinyphysics_core.simulator import TinyPhysicsSimulator

sns.set_theme()
signal.signal(signal.SIGINT, signal.SIG_DFL)  # Enable Ctrl-C on plot windows

_MODEL_CACHE: Dict[Tuple[str, bool], TinyPhysicsModel] = {}


def _get_or_create_model(model_path: Union[str, Path], debug: bool) -> TinyPhysicsModel:
  key = (str(Path(model_path).resolve()), debug)
  if key not in _MODEL_CACHE:
    _MODEL_CACHE[key] = TinyPhysicsModel(str(model_path), debug=debug)
  else:
    _MODEL_CACHE[key].reset()
  return _MODEL_CACHE[key]


def get_available_controllers() -> List[str]:
  """Discover controllers packaged with the repository."""
  controllers_dir = BASE_DIR / "controllers"
  return [f.stem for f in controllers_dir.iterdir() if f.is_file() and f.suffix == '.py' and f.stem != '__init__']


def run_rollout(data_path: Union[str, Path], controller_type: str, model_path: Union[str, Path], debug: bool = False):
  model = _get_or_create_model(model_path, debug=debug)
  controller = importlib.import_module(f'controllers.{controller_type}').Controller()
  controller.reset()
  simulator = TinyPhysicsSimulator(model, str(data_path), controller=controller)
  runner = RolloutRunner(simulator, debug=debug)
  result = runner.run()
  controller.reset()
  return result


def download_dataset() -> None:
  print("Downloading dataset (0.6G)...")
  DATASET_PATH.mkdir(parents=True, exist_ok=True)
  with urllib.request.urlopen(DATASET_URL) as resp:
    with zipfile.ZipFile(BytesIO(resp.read())) as z:
      for member in z.namelist():
        if not member.endswith('/'):
          with z.open(member) as src, open(DATASET_PATH / os.path.basename(member), 'wb') as dest:  # noqa: PL134
            dest.write(src.read())


def _summarize_costs(costs_df: pd.DataFrame) -> None:
  print(f"\nAverage lataccel_cost: {np.mean(costs_df['lataccel_cost']):>6.4}, average jerk_cost: {np.mean(costs_df['jerk_cost']):>6.4}, average total_cost: {np.mean(costs_df['total_cost']):>6.4}")
  for cost in costs_df.columns:
    plt.hist(costs_df[cost], bins=HISTOGRAM_BINS, label=cost, alpha=0.5)
  plt.xlabel('costs')
  plt.ylabel('Frequency')
  plt.title('Cost Distribution')
  plt.legend()
  plt.show()


if __name__ == "__main__":
  available_controllers = get_available_controllers()
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
  parser.add_argument("--data_path", type=str, default=str(DATASET_PATH))
  parser.add_argument("--num_segs", type=int, default=1)
  parser.add_argument("--debug", action='store_true')
  parser.add_argument("--controller", default='pid', choices=available_controllers)
  args = parser.parse_args()

  if not DATASET_PATH.exists():
    download_dataset()

  data_path = Path(args.data_path)
  if data_path.is_file():
    result = run_rollout(data_path, args.controller, args.model_path, debug=args.debug)
    print(f"\nAverage lataccel_cost: {result.cost['lataccel_cost']:>6.4}, average jerk_cost: {result.cost['jerk_cost']:>6.4}, average total_cost: {result.cost['total_cost']:>6.4}")
  elif data_path.is_dir():
    run_rollout_partial = partial(run_rollout, controller_type=args.controller, model_path=args.model_path, debug=False)
    files = sorted(data_path.iterdir())[:args.num_segs]
    results = process_map(run_rollout_partial, files, max_workers=16, chunksize=10)
    costs = [result.cost for result in results]
    costs_df = pd.DataFrame(costs)
    _summarize_costs(costs_df)
  else:
    raise FileNotFoundError(f"data_path does not exist: {data_path}")
