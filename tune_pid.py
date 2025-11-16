import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tinyphysics import get_available_controllers
from tinyphysics_core.config import (
  CONTROL_START_IDX,
  DATASET_PATH,
  DEFAULT_MODEL_PATH,
  DEL_T,
)
from tinyphysics_core.model import TinyPhysicsModel
from tinyphysics_core.runner import RolloutRunner
from tinyphysics_core.simulator import TinyPhysicsSimulator


@dataclass
class GainSearchResult:
  gain: float
  oscillating: bool
  period: Optional[float]
  zero_crossings: int
  amplitude: float
  std: float
  max_error: float
  cost: Dict[str, float]


def _load_controller(controller_type: str) -> type:
  module = __import__(f"controllers.{controller_type}", fromlist=["Controller"])
  return module.Controller


def run_rollout_with_gains(
  model: TinyPhysicsModel,
  segment_path: Path,
  controller_cls: type,
  p_gain: float,
  i_gain: float,
  d_gain: float
):
  """Execute a rollout with the specified PID gains and return the RolloutResult."""
  model.reset()
  controller = controller_cls(p=p_gain, i=i_gain, d=d_gain)
  simulator = TinyPhysicsSimulator(model, str(segment_path), controller=controller)
  runner = RolloutRunner(simulator, debug=False)
  return runner.run()


def _zero_crossing_steps(series: np.ndarray, start_step: int) -> List[int]:
  signs = np.sign(series)
  signs[signs == 0] = 1
  crossings = np.where(signs[:-1] * signs[1:] < 0)[0]
  return [start_step + idx + 1 for idx in crossings]


def analyze_gain_behavior(
  errors: np.ndarray,
  steps: np.ndarray,
  min_crossings: int,
  min_amplitude: float
) -> Tuple[bool, Optional[float], int, float, float, float]:
  """Inspect the error signal to determine whether it is in sustained oscillation."""
  control_errors = errors[CONTROL_START_IDX:]
  control_steps = steps[CONTROL_START_IDX:]
  if control_errors.size == 0:
    return False, None, 0, 0.0, 0.0, 0.0
  tail_start = control_errors.size // 3
  tail_errors = control_errors[tail_start:]
  tail_steps_start = int(control_steps[tail_start])
  zero_cross_steps = _zero_crossing_steps(tail_errors, tail_steps_start)
  amplitude = float(np.max(tail_errors) - np.min(tail_errors))
  std = float(np.std(tail_errors))
  max_error = float(np.max(np.abs(tail_errors)))
  oscillating = len(zero_cross_steps) >= min_crossings and amplitude >= min_amplitude
  period = None
  if len(zero_cross_steps) >= 4:
    half_period_steps = np.diff(zero_cross_steps)
    if half_period_steps.size > 0:
      period = float(np.mean(half_period_steps) * 2 * DEL_T)
  return oscillating, period, len(zero_cross_steps), amplitude, std, max_error


def evaluate_gain(
  model: TinyPhysicsModel,
  segment_path: Path,
  controller_cls: type,
  gain: float,
  min_crossings: int,
  min_amplitude: float
) -> GainSearchResult:
  """Run a rollout with only proportional gain and report whether it oscillates."""
  result = run_rollout_with_gains(model, segment_path, controller_cls, p_gain=gain, i_gain=0.0, d_gain=0.0)
  target = np.array(result.target_lataccel_history, dtype=float)
  actual = np.array(result.current_lataccel_history, dtype=float)
  steps = np.arange(len(target))
  errors = target - actual
  oscillating, period, crossings, amplitude, std, max_error = analyze_gain_behavior(
    errors, steps, min_crossings=min_crossings, min_amplitude=min_amplitude
  )
  return GainSearchResult(
    gain=gain,
    oscillating=oscillating,
    period=period,
    zero_crossings=crossings,
    amplitude=amplitude,
    std=std,
    max_error=max_error,
    cost=result.cost
  )


def tune_ultimate_gain(
  model: TinyPhysicsModel,
  segment_path: Path,
  controller_cls: type,
  start_gain: float,
  max_gain: float,
  growth_factor: float,
  refine_steps: int,
  min_crossings: int,
  min_amplitude: float
) -> Tuple[GainSearchResult, List[GainSearchResult]]:
  """Search for the ultimate gain Ku using a coarse sweep plus binary refinement."""
  history: List[GainSearchResult] = []
  gain = start_gain
  last_stable: Optional[GainSearchResult] = None
  first_oscillation: Optional[GainSearchResult] = None

  while gain <= max_gain:
    result = evaluate_gain(model, segment_path, controller_cls, gain, min_crossings, min_amplitude)
    history.append(result)
    if result.oscillating:
      first_oscillation = result
      break
    last_stable = result
    gain *= growth_factor

  if first_oscillation is None:
    raise RuntimeError("Failed to induce sustained oscillation within the provided gain range.")
  if last_stable is None:
    last_stable = first_oscillation

  low = last_stable.gain
  high = first_oscillation.gain
  high_result = first_oscillation

  for _ in range(refine_steps):
    mid = (low + high) / 2.0
    result = evaluate_gain(model, segment_path, controller_cls, mid, min_crossings, min_amplitude)
    history.append(result)
    if result.oscillating:
      high = mid
      high_result = result
    else:
      low = mid
      last_stable = result

  if high_result.period is None:
    raise RuntimeError("Unable to measure oscillation period; consider adjusting min_crossings or dataset segment.")
  return high_result, history


def ziegler_nichols_pid(ku: float, pu: float) -> Tuple[float, float, float]:
  """Return PID gains based on the Ziegler-Nichols closed-loop tuning rules."""
  kp = 0.6 * ku
  ki_continuous = 2 * kp / pu
  kd_continuous = kp * pu / 8
  ki = ki_continuous * DEL_T
  kd = -(kd_continuous / DEL_T)
  return kp, ki, kd


def resolve_segment_path(data_path: Path, segment_path: Optional[Path], segment_index: int) -> Path:
  if segment_path is not None:
    if not segment_path.exists():
      raise FileNotFoundError(f"segment_path does not exist: {segment_path}")
    return segment_path
  files = sorted(data_path.glob("*.csv"))
  if not files:
    raise FileNotFoundError(f"No CSV files found under {data_path}")
  if segment_index < 0 or segment_index >= len(files):
    raise IndexError(f"segment_index {segment_index} out of range (found {len(files)} files)")
  return files[segment_index]


def verify_gains(model: TinyPhysicsModel, segment_path: Path, controller_cls: type, kp: float, ki: float, kd: float) -> Dict[str, float]:
  result = run_rollout_with_gains(model, segment_path, controller_cls, p_gain=kp, i_gain=ki, d_gain=kd)
  return result.cost


def main() -> None:
  parser = argparse.ArgumentParser(description="Tune PID gains via Ziegler-Nichols on a single segment.")
  parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the ONNX physics model.")
  parser.add_argument("--data_path", type=Path, default=DATASET_PATH, help="Directory containing segment CSV files.")
  parser.add_argument("--segment_path", type=Path, default=None, help="Optional explicit segment CSV path.")
  parser.add_argument("--segment_index", type=int, default=0, help="Index of the CSV file inside data_path to use.")
  parser.add_argument("--controller", default="pid_opt", choices=get_available_controllers(), help="Controller module to tune.")
  parser.add_argument("--start_gain", type=float, default=0.05, help="Initial proportional gain for the sweep.")
  parser.add_argument("--max_gain", type=float, default=1.0, help="Upper bound for the proportional gain sweep.")
  parser.add_argument("--growth_factor", type=float, default=1.25, help="Multiplier applied after each sweep iteration.")
  parser.add_argument("--refine_steps", type=int, default=5, help="Binary search steps to refine the ultimate gain.")
  parser.add_argument("--min_crossings", type=int, default=15, help="Minimum zero crossings to consider the signal oscillatory.")
  parser.add_argument("--min_amplitude", type=float, default=0.35, help="Minimum error amplitude for oscillation detection.")
  parser.add_argument("--verify", action="store_true", help="Run a rollout with the tuned gains to report cost.")
  args = parser.parse_args()

  data_path = args.data_path
  data_path.mkdir(parents=True, exist_ok=True)
  segment_path = resolve_segment_path(data_path, args.segment_path, args.segment_index)
  model = TinyPhysicsModel(str(args.model_path))
  controller_cls = _load_controller(args.controller)

  print(f"Tuning PID gains using segment: {segment_path}")
  ultimate_result, history = tune_ultimate_gain(
    model=model,
    segment_path=segment_path,
    controller_cls=controller_cls,
    start_gain=args.start_gain,
    max_gain=args.max_gain,
    growth_factor=args.growth_factor,
    refine_steps=args.refine_steps,
    min_crossings=args.min_crossings,
    min_amplitude=args.min_amplitude
  )

  ku = ultimate_result.gain
  pu = ultimate_result.period
  kp, ki, kd = ziegler_nichols_pid(ku, pu)

  print("\n=== Gain Sweep Summary ===")
  for entry in history:
    status = "OSC" if entry.oscillating else "stable"
    print(
      f"P={entry.gain:.5f} -> {status} | "
      f"crossings={entry.zero_crossings:3d}, amp={entry.amplitude:.3f}, std={entry.std:.3f}, "
      f"max_err={entry.max_error:.3f}, total_cost={entry.cost['total_cost']:.3f}"
    )

  print("\n=== Ziegler-Nichols Results ===")
  print(f"Ultimate gain Ku: {ku:.6f}")
  print(f"Ultimate period Pu: {pu:.6f} s")
  print(f"Recommended PID gains -> P: {kp:.6f}, I: {ki:.6f}, D: {kd:.6f}")

  if args.verify:
    costs = verify_gains(model, segment_path, controller_cls, kp, ki, kd)
    print("\nVerification rollout cost:")
    for key, value in costs.items():
      print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
  main()
