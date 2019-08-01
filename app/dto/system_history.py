from typing import Dict, List, NamedTuple
from singleton_decorator import singleton

from .state import State

@singleton
class SystemHistory(NamedTuple):
  totalPower: List[float] = []
  frequency: List[float] = []
  generators: Dict[str, List[float]] = {}
  loads: Dict[str, List[float]] = {}
  actualCosts: Dict[str, List[float]] = {}
  optimalCosts: Dict[str, List[float]] = {}
  totalCosts: Dict[str, List[float]] = {}

  def pushState(self, state: State):
    self.totalPower.append(state.totalPower)
    self.frequency.append(state.frequency)
    for load in state.loads:
      self.loads.setdefault(load.id_, []).append(load.power)
    for gen in state.generators:
      self.generators.setdefault(gen.id_, []).append(gen.power)
    for costItem in state.actualCost:
      self.actualCosts.setdefault(costItem.id_, []).append(costItem.cost)
    for costItem in state.optimalCost:
      self.optimalCosts.setdefault(costItem.id_, []).append(costItem.cost)
    for totalCostItem in state.totalCost:
      self.totalCosts.setdefault(totalCostItem.id_, []).append(totalCostItem.cost)
