from abc import ABC, abstractmethod
from typing import Any


class BaseController(ABC):
  """Common interface for every feedback controller used by the simulator."""

  @abstractmethod
  def update(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any) -> float:
    """
    Args:
      target_lataccel: Desired lateral acceleration.
      current_lataccel: Current lateral acceleration reported by the simulator.
      state: Vehicle state snapshot for the current timestep.
      future_plan: Planned trajectory (lataccel, roll_lataccel, v_ego, a_ego).
    Returns:
      Steering command to be applied to the vehicle.
    """

  def reset(self) -> None:
    """Optional hook for controllers keeping internal state (integrators, etc.)."""
    return None
