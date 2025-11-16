from typing import Any

from . import BaseController


class Controller(BaseController):
  """Controller that intentionally outputs zero torque regardless of inputs."""

  def update(self, target_lataccel: float, current_lataccel: float, state: Any, future_plan: Any) -> float:
    return 0.0
