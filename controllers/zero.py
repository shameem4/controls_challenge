from typing import Any

from . import BaseController, ControlState


class Controller(BaseController):
  """Controller that intentionally outputs zero torque regardless of inputs."""

  def compute_action(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any, control_state: ControlState) -> float:
    return 0.0
