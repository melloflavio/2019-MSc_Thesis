from typing import List
import pydash as _

from dto import NodeStateCost, ElectricalState, SystemHistory, NodePowerUpdate

from .area_dynamics import AreaDynamics
from .cost_calculator import CostCalculator

class ElectricalSystem:
  def __init__(self, initialFrequency, loads, generators):
    self.loads = loads
    self.generators = generators

    initialPower = sum([gen.getOutput() for gen in self.generators])
    minCost, minCostNodesPower, minCostNodesCost = CostCalculator.calculateMinimumCost(self.generators, initialPower)

    SystemHistory().pushState(ElectricalState(
        totalPower=initialPower,
        totalLoad=sum([l.getLoad() for l in self.loads]),
        frequency=initialFrequency,
        loads=[l.toNodeStatePower() for l in self.loads],
        generators=[g.toNodeStatePower() for g in self.generators],
        actualCost=[g.toNodeStateCost() for g in self.generators],
        costOptimalCost=minCostNodesCost,
        costOptimalPower=minCostNodesPower,
        totalCost=[
            NodeStateCost(id_= "Minimum", cost=minCost),
            NodeStateCost(id_= "Actual", cost=sum(g.getCost() for g in self.generators)),
            ]
        ))

  def updateGenerators(self, generatorsUpdates: List[NodePowerUpdate]):
    # 1. Update the power output for each generator
    for generatorUpdate in generatorsUpdates:
      selectedGenerator = _.find(self.generators, lambda gen: gen.getId() == generatorUpdate.id_)
      selectedGenerator.updateOutput(generatorUpdate.deltaPower)

    # 2. Calculate the next total power output
    zg = sum([gen.power for gen in generatorsUpdates]) # Total Secondary Action (Z) from all generators
    totalPowerOld = SystemHistory().totalPower[-1] # Old Total Power (P_G_Old)
    frequencyOld = SystemHistory().frequency[-1]
    powerGeneratedNew = AreaDynamics.calculatePowerGeneratedNew(zg, totalPowerOld, frequencyOld)

    # 3. Calculate the next frequency
    totalLoad = sum([l.getLoad() for l in self.loads])
    frequencyNew = AreaDynamics.calculateFrequencyNew(powerGeneratedNew, totalLoad, frequencyOld)

    # 4. Calculate the optimal generation cost of the current output
    minCost, minCostNodesPower, minCostNodesCost = CostCalculator.calculateMinimumCost(self.generators, zg)

    # 5. Push the new state to system history
    SystemHistory().pushState(ElectricalState(
        totalPower=powerGeneratedNew,
        totalLoad=sum([l.getLoad() for l in self.loads]),
        frequency=frequencyNew,
        loads=[l.toNodeStatePower() for l in self.loads],
        generators=[g.toNodeStatePower() for g in self.generators],
        actualCost=[g.toNodeStateCost() for g in self.generators],
        costOptimalCost=minCostNodesCost,
        costOptimalPower=minCostNodesPower,
        totalCost=[
            NodeStateCost(id_="Minimum", cost=minCost),
            NodeStateCost(id_="Actual", cost=sum(g.getCost() for g in self.generators)),
            ]
        ))

  def getGeneratorIds(self):
    return [g.getId() for g in self.generators]

  def getCurrentDeltaF(self):
    currentFrequency = ElectricalState().frequency[-1]
    return AreaDynamics.getDeltaFrequency(currentFrequency)
