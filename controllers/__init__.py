from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional


class ControlPhase(Enum):
  """Phases for the controller state machine."""
  WARMUP = auto()
  ACTIVE = auto()


@dataclass(frozen=True)
class ControlState:
  """State provided to controllers every simulator step."""
  phase: ControlPhase
  step_idx: int
  forced_action: Optional[float] = None
  previous_phase: ControlPhase = ControlPhase.WARMUP

  @property
  def is_control_active(self) -> bool:
    return self.phase is ControlPhase.ACTIVE

  @property
  def just_activated(self) -> bool:
    return self.previous_phase is ControlPhase.WARMUP and self.phase is ControlPhase.ACTIVE


class BaseController(ABC):
  """Common interface for every feedback controller used by the simulator."""

  @abstractmethod
  def update(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any, control_state: ControlState) -> float:
    """
    Args:
      target_lataccel: Desired lateral acceleration.
      current_lataccel: Current lateral acceleration reported by the simulator.
      state: Vehicle state snapshot for the current timestep.
      future_plan: Planned trajectory (lataccel, roll_lataccel, v_ego, a_ego).
      control_state: Metadata describing the controller's current phase and forced action.
    Returns:
      Steering command to be applied to the vehicle.
    """

  def reset(self) -> None:
    """Optional hook for controllers keeping internal state (integrators, etc.)."""
    return None
