from typing import Dict, List, NamedTuple
from singleton_decorator import singleton

from .state import State

@singleton
class SystemHistory(NamedTuple):
  totalPower: List[float] = []
  frequency: List[float] = []
  generators: Dict[str, List[float]] = {}
  loads: Dict[str, List[float]] = {}

  def pushState(self, state: State):
    self.totalPower.append(state.totalPower)
    self.frequency.append(state.frequency)
    for load in state.loads:
      self.loads.setdefault(load.id_, []).append(load.power)
    for gen in state.generators:
      self.generators.setdefault(gen.id_, []).append(gen.power)
