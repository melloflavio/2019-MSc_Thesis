from typing import Dict, List, NamedTuple
from singleton_decorator import singleton

from .state import ElectricalState

@singleton
class SystemHistory(NamedTuple):
  steps: int = []
  totalPower: List[float] = []
  totalLoad: List[float] = []
  frequency: List[float] = []
  generators: Dict[str, List[float]] = {}
  loads: Dict[str, List[float]] = {}
  actualCosts: Dict[str, List[float]] = {}
  costOptimalCosts: Dict[str, List[float]] = {}
  costOptimalPowers: Dict[str, List[float]] = {}
  totalCosts: Dict[str, List[float]] = {}

  def pushState(self, state: ElectricalState):
    self.totalPower.append(state.totalPower)
    self.totalLoad.append(state.totalLoad)
    self.frequency.append(state.frequency)
    for load in state.loads:
      self.loads.setdefault(load.id_, []).append(load.power)
    for gen in state.generators:
      self.generators.setdefault(gen.id_, []).append(gen.power)
    for costItem in state.actualCost:
      self.actualCosts.setdefault(costItem.id_, []).append(costItem.cost)
    for costItem in state.costOptimalCost:
      self.costOptimalCosts.setdefault(costItem.id_, []).append(costItem.cost)
    for powerItem in state.costOptimalPower:
      self.costOptimalPowers.setdefault(powerItem.id_, []).append(powerItem.power)
    for totalCostItem in state.totalCost:
      self.totalCosts.setdefault(totalCostItem.id_, []).append(totalCostItem.cost)
    self.steps.append(len(self.steps))
