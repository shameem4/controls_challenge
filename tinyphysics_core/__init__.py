"""Core modules that power the tinyphysics CLI."""

from .config import *  # re-export constants for convenience

_LAZY_EXPORTS = [
  "BasePhysicsModel",
  "TinyPhysicsModel",
  "RolloutRunner",
  "TinyPhysicsSimulator",
  "FuturePlan",
  "RolloutResult",
  "SimulationHistories",
  "State",
]

__all__ = [name for name in dir() if name.isupper()] + _LAZY_EXPORTS


def __getattr__(name):
  if name in {"BasePhysicsModel", "TinyPhysicsModel"}:
    from .model import BasePhysicsModel, TinyPhysicsModel
    return {"BasePhysicsModel": BasePhysicsModel, "TinyPhysicsModel": TinyPhysicsModel}[name]
  if name in {"RolloutRunner"}:
    from .runner import RolloutRunner
    return RolloutRunner
  if name in {"TinyPhysicsSimulator"}:
    from .simulator import TinyPhysicsSimulator
    return TinyPhysicsSimulator
  if name in {"FuturePlan", "RolloutResult", "SimulationHistories", "State"}:
    from .types import FuturePlan, RolloutResult, SimulationHistories, State
    return {
      "FuturePlan": FuturePlan,
      "RolloutResult": RolloutResult,
      "SimulationHistories": SimulationHistories,
      "State": State
    }[name]
  raise AttributeError(f"module 'tinyphysics_core' has no attribute '{name}'")
