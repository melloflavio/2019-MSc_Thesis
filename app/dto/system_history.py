from typing import Dict, List, NamedTuple
from singleton_decorator import singleton

@singleton
class SystemHistory(NamedTuple):
  totalPower: List[float] = []
  frequency: List[float] = []
  generators: Dict[str, List[float]] = {}
  loads: Dict[str, List[float]] = {}

  def pushState(self, totalPower: float, frequency: float, loads: List[any], generators: List[any]):
    self.totalPower.append(totalPower)
    self.frequency.append(frequency)
    for load in loads:
      self.loads.setdefault(load.getId(), []).append(load.getLoad())
    for gen in generators:
      self.generators.setdefault(gen.getId(), []).append(gen.getOutput())
