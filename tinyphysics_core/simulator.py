from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from controllers import BaseController, ControlPhase, ControlState
from tinyphysics_core.controller_storage import ControllerHistory
from .config import (
  ACC_G,
  CONTROL_START_IDX,
  CONTEXT_LENGTH,
  COST_END_IDX,
  DEL_T,
  FUTURE_PLAN_STEPS,
  LAT_ACCEL_COST_MULTIPLIER,
  MAX_ACC_DELTA,
  STEER_RANGE,
)
from .model import BasePhysicsModel
from .types import FuturePlan, RolloutResult, SimulationHistories, State
from .utils import compute_segment_seed


class TinyPhysicsSimulator:
  """Pairs a physics model with a controller to roll out trajectories."""

  def __init__(self, model: BasePhysicsModel, data_path: Union[str, Path], controller: BaseController) -> None:
    self.data_path = Path(data_path)
    self.sim_model = model
    self.controller = controller
    self.controller_history = ControllerHistory()
    self._cost_histories: Dict[str, List[float]] = {}
    self.data = self.get_data(self.data_path)
    self.history: SimulationHistories
    self.current_lataccel: float = 0.0
    self._control_phase = ControlPhase.WARMUP
    self.reset()

  @property
  def state_history(self):
    return self.history.states

  @property
  def target_lataccel_history(self):
    return self.history.target_lataccel

  @property
  def current_lataccel_history(self):
    return self.history.current_lataccel

  @property
  def action_history(self):
    return self.history.actions

  @property
  def cost_histories(self) -> Dict[str, List[float]]:
    return self._cost_histories

  def reset(self) -> None:
    self.step_idx = CONTEXT_LENGTH
    bootstrap_data = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
    initial_actions = self.data['steer_command'].values[:self.step_idx].tolist()
    self.history = SimulationHistories.bootstrap(bootstrap_data, initial_actions)
    self.current_lataccel = self.current_lataccel_history[-1]
    self._seed_random()
    self.controller.reset()
    self.controller_history.reset()
    self._init_cost_histories()
    self._control_phase = ControlPhase.WARMUP

  def _init_cost_histories(self) -> None:
    self._cost_histories = {
      'step_idx': [],
      'lataccel_cost': [],
      'jerk_cost': [],
      'total_cost': []
    }

  def _seed_random(self) -> None:
    seed = compute_segment_seed(self.data, identifier=self.data_path.name)
    np.random.seed(seed)
    self.sim_model.seed(seed)

  def get_data(self, data_path: Path) -> pd.DataFrame:
    """Load and preprocess a driving segment."""
    df = pd.read_csv(data_path)
    processed_df = pd.DataFrame({
      'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
      'v_ego': df['vEgo'].values,
      'a_ego': df['aEgo'].values,
      'target_lataccel': df['targetLateralAcceleration'].values,
      'steer_command': -df['steerCommand'].values  # steer commands logged left-positive, simulator assumes right-positive
    })
    return processed_df

  def sim_step(self, step_idx: int) -> float:
    pred = self.sim_model.predict_lataccel(
      sim_states=self.history.recent_states(CONTEXT_LENGTH),
      actions=self.history.recent_actions(CONTEXT_LENGTH),
      past_preds=self.history.recent_lataccels(CONTEXT_LENGTH)
    )
    pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
    if step_idx >= CONTROL_START_IDX:
      self.current_lataccel = pred
    else:
      self.current_lataccel = self.get_state_target_futureplan(step_idx)[1]
    self.history.append_current_lataccel(self.current_lataccel)
    return pred

  def _record_step_cost(self, step_idx: int) -> Dict[str, float] | None:
    if step_idx < CONTROL_START_IDX:
      return None
    if COST_END_IDX is not None and step_idx >= COST_END_IDX:
      return None
    if not self.history.current_lataccel or not self.history.target_lataccel:
      return None

    target = float(self.history.target_lataccel[-1])
    pred = float(self.history.current_lataccel[-1])
    lataccel_cost = ((target - pred) ** 2) * 100.0

    jerk_cost = 0.0
    if len(self.history.current_lataccel) >= 2:
      prev_pred = float(self.history.current_lataccel[-2])
      jerk_cost = (((pred - prev_pred) / DEL_T) ** 2) * 100.0

    total_cost = (lataccel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    self._cost_histories['step_idx'].append(step_idx)
    self._cost_histories['lataccel_cost'].append(lataccel_cost)
    self._cost_histories['jerk_cost'].append(jerk_cost)
    self._cost_histories['total_cost'].append(total_cost)
    metrics = {
      'step_idx': step_idx,
      'lataccel_cost': lataccel_cost,
      'jerk_cost': jerk_cost,
      'total_cost': total_cost
    }
    self.controller_history.append('step_cost', metrics)
    return metrics

  def control_step(self, step_idx: int, target_lataccel: float, state: State, futureplan: FuturePlan) -> float:
    new_phase = ControlPhase.ACTIVE if step_idx >= CONTROL_START_IDX else ControlPhase.WARMUP
    forced_action = float(self.data['steer_command'].values[step_idx]) if new_phase is ControlPhase.WARMUP else None
    control_state = ControlState(
      phase=new_phase,
      previous_phase=self._control_phase,
      step_idx=step_idx,
      forced_action=forced_action,
      history=self.controller_history
    )
    self._control_phase = new_phase
    action = self.controller.update(
      target_lataccel,
      self.current_lataccel,
      state,
      future_plan=futureplan,
      control_state=control_state
    )
    if new_phase is ControlPhase.WARMUP:
      return (forced_action if forced_action is not None else 0.0), control_state
    return float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1])), control_state

  def get_state_target_futureplan(self, step_idx: int) -> Tuple[State, float, FuturePlan]:
    state = self.data.iloc[step_idx]
    end_idx = step_idx + 1 + FUTURE_PLAN_STEPS
    target_series = self.data['target_lataccel'].values
    roll_series = self.data['roll_lataccel'].values
    v_series = self.data['v_ego'].values
    a_series = self.data['a_ego'].values
    return (
      State(roll_lataccel=state['roll_lataccel'], v_ego=state['v_ego'], a_ego=state['a_ego']),
      state['target_lataccel'],
      FuturePlan(
        lataccel=target_series[step_idx + 1:end_idx].tolist(),
        roll_lataccel=roll_series[step_idx + 1:end_idx].tolist(),
        v_ego=v_series[step_idx + 1:end_idx].tolist(),
        a_ego=a_series[step_idx + 1:end_idx].tolist()
      )
    )

  def step(self) -> None:
    state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
    self.history.append_state(state, target)
    action, control_state = self.control_step(self.step_idx, target, state, futureplan)
    self.history.append_action(action)
    predicted_lataccel = self.sim_step(self.step_idx)
    self.controller_history.append('predicted_lataccel', predicted_lataccel)
    step_metrics = self._record_step_cost(self.step_idx)
    self.controller.on_simulation_update(predicted_lataccel, control_state, step_metrics=step_metrics)
    self.step_idx += 1

  def compute_avg_cost(self) -> Dict[str, float]:
    """Return the mean of the recorded per-step costs."""
    def _mean(values: List[float]) -> float:
      return float(np.mean(values)) if values else 0.0

    lataccel_cost = _mean(self._cost_histories.get('lataccel_cost', []))
    jerk_cost = _mean(self._cost_histories.get('jerk_cost', []))
    total_cost = _mean(self._cost_histories.get('total_cost', []))
    return {'lataccel_cost': lataccel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost}

  def build_rollout_result(self) -> RolloutResult:
    return RolloutResult(
      cost=self.compute_avg_cost(),
      cost_history={name: list(values) for name, values in self._cost_histories.items()},
      target_lataccel_history=list(self.target_lataccel_history),
      current_lataccel_history=list(self.current_lataccel_history),
      action_history=list(self.action_history),
      diagnostics=self.controller.get_diagnostics(),
      state_stats=self._compute_state_stats()
    )

  @property
  def total_steps(self) -> int:
    return len(self.data)

  def _compute_state_stats(self) -> Dict[str, float]:
    states = self.state_history
    v_ego = np.array([s.v_ego for s in states])
    a_ego = np.array([s.a_ego for s in states])
    roll_lataccel = np.array([s.roll_lataccel for s in states])
    target_lataccel = np.array(self.target_lataccel_history)
    return {
      'avg_v_ego': float(np.mean(v_ego)),
      'max_v_ego': float(np.max(v_ego)),
      'avg_abs_roll_lataccel': float(np.mean(np.abs(roll_lataccel))),
      'avg_abs_target_lataccel': float(np.mean(np.abs(target_lataccel))),
      'avg_abs_a_ego': float(np.mean(np.abs(a_ego)))
    }
