from dataclasses import dataclass, field
from typing import Dict, List

from .electrical_state import ElectricalState

@dataclass
class SystemHistory:
  steps: int = field(default_factory=list)
  totalPower: List[float] = field(default_factory=list)
  totalLoad: List[float] = field(default_factory=list)
  frequency: List[float] = field(default_factory=list)
  generators: Dict[str, List[float]] = field(default_factory=dict)
  loads: Dict[str, List[float]] = field(default_factory=dict)
  actualCosts: Dict[str, List[float]] = field(default_factory=dict)
  costOptimalCosts: Dict[str, List[float]] = field(default_factory=dict)
  costOptimalPowers: Dict[str, List[float]] = field(default_factory=dict)
  totalCosts: Dict[str, List[float]] = field(default_factory=dict)

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
