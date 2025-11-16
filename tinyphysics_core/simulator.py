from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

from controllers import BaseController, ControlPhase, ControlState
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

  def reset(self) -> None:
    self.step_idx = CONTEXT_LENGTH
    bootstrap_data = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
    initial_actions = self.data['steer_command'].values[:self.step_idx].tolist()
    self.history = SimulationHistories.bootstrap(bootstrap_data, initial_actions)
    self.current_lataccel = self.current_lataccel_history[-1]
    self._seed_random()
    self.controller.reset()
    self._control_phase = ControlPhase.WARMUP

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

  def sim_step(self, step_idx: int) -> None:
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

  def control_step(self, step_idx: int, target_lataccel: float, state: State, futureplan: FuturePlan) -> float:
    new_phase = ControlPhase.ACTIVE if step_idx >= CONTROL_START_IDX else ControlPhase.WARMUP
    forced_action = float(self.data['steer_command'].values[step_idx]) if new_phase is ControlPhase.WARMUP else None
    control_state = ControlState(
      phase=new_phase,
      previous_phase=self._control_phase,
      step_idx=step_idx,
      forced_action=forced_action
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
      return forced_action if forced_action is not None else 0.0
    return float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))

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
    action = self.control_step(self.step_idx, target, state, futureplan)
    self.history.append_action(action)
    self.sim_step(self.step_idx)
    self.step_idx += 1

  def compute_cost(self):
    target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
    pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]

    lat_accel_cost = np.mean((target - pred)**2) * 100
    jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    return {'lataccel_cost': lat_accel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost}

  def build_rollout_result(self) -> RolloutResult:
    return RolloutResult(
      cost=self.compute_cost(),
      target_lataccel_history=list(self.target_lataccel_history),
      current_lataccel_history=list(self.current_lataccel_history)
    )

  @property
  def total_steps(self) -> int:
    return len(self.data)
