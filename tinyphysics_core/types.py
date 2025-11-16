from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class State:
  roll_lataccel: float
  v_ego: float
  a_ego: float


@dataclass(frozen=True)
class FuturePlan:
  lataccel: List[float]
  roll_lataccel: List[float]
  v_ego: List[float]
  a_ego: List[float]


@dataclass
class SimulationHistories:
  """Container for the time-series data produced while simulating a segment."""
  states: List[State]
  actions: List[float]
  current_lataccel: List[float]
  target_lataccel: List[float]

  @classmethod
  def bootstrap(cls, bootstrap_data: List[Tuple[State, float, FuturePlan]], initial_actions: List[float]) -> "SimulationHistories":
    states = [entry[0] for entry in bootstrap_data]
    target_lataccel = [entry[1] for entry in bootstrap_data]
    return cls(
      states=states,
      actions=list(initial_actions),
      current_lataccel=list(target_lataccel),
      target_lataccel=target_lataccel
    )

  def append_state(self, state: State, target_lataccel: float) -> None:
    self.states.append(state)
    self.target_lataccel.append(target_lataccel)

  def append_action(self, action: float) -> None:
    self.actions.append(action)

  def append_current_lataccel(self, lataccel: float) -> None:
    self.current_lataccel.append(lataccel)

  def recent_states(self, context_length: int) -> List[State]:
    return self.states[-context_length:]

  def recent_actions(self, context_length: int) -> List[float]:
    return self.actions[-context_length:]

  def recent_lataccels(self, context_length: int) -> List[float]:
    return self.current_lataccel[-context_length:]


@dataclass
class RolloutResult:
  cost: Dict[str, float]
  target_lataccel_history: List[float]
  current_lataccel_history: List[float]
  action_history: List[float]
  diagnostics: Dict[str, float]
  state_stats: Dict[str, float]

  def as_tuple(self):
    return self.cost, self.target_lataccel_history, self.current_lataccel_history
