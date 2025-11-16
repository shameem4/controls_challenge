from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ControllerHistory:
  """Generic history tracker controllers can use to store arbitrary sequences."""
  _storage: Dict[str, List[Any]] = field(default_factory=dict)

  def register(self, name: str) -> None:
    self._storage.setdefault(name, [])

  def append(self, name: str, value: Any) -> None:
    self.register(name)
    self._storage[name].append(value)

  def recent(self, name: str, length: int) -> List[Any]:
    values = self._storage.get(name, [])
    if length <= 0:
      return []
    return values[-length:]

  def latest(self, name: str) -> Optional[Any]:
    values = self._storage.get(name, [])
    return values[-1] if values else None

  def reset(self) -> None:
    for series in self._storage.values():
      series.clear()
