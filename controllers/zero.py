from typing import Any

from . import BaseController, ControlPhase, ControlState


class Controller(BaseController):
  """Controller that intentionally outputs zero torque regardless of inputs."""

  def update(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any, control_state: ControlState) -> float:
    if control_state.phase is ControlPhase.WARMUP and control_state.forced_action is not None:
      return control_state.forced_action
    return 0.0
