from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from tinyphysics_core.controller_storage import ControllerHistory


class ControlPhase(Enum):
  """Phases for the controller state machine."""
  WARMUP = auto()
  ACTIVE = auto()


@dataclass
class ControlState:
  """State metadata shared with controllers each timestep."""
  phase: ControlPhase
  step_idx: int
  forced_action: Optional[float] = None
  previous_phase: ControlPhase = ControlPhase.WARMUP
  history: ControllerHistory = field(default_factory=ControllerHistory)
  reset_requested: bool = False

  @property
  def is_control_active(self) -> bool:
    return self.phase is ControlPhase.ACTIVE

  @property
  def just_activated(self) -> bool:
    return self.previous_phase is ControlPhase.WARMUP and self.phase is ControlPhase.ACTIVE


class BaseController(ABC):
  """Common interface for every feedback controller used by the simulator."""

  def update(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any, control_state: ControlState) -> float:
    if control_state.phase is ControlPhase.WARMUP:
      return control_state.forced_action if control_state.forced_action is not None else 0.0
    if control_state.just_activated or control_state.reset_requested:
      self.reset()
    return self.compute_action(target_lataccel, current_lataccel, state, future_plan, control_state)

  @abstractmethod
  def compute_action(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any, control_state: ControlState) -> float:
    """Implement controller-specific logic."""

  def reset(self) -> None:
    """Optional hook for controllers keeping internal state (integrators, etc.)."""
    return None
