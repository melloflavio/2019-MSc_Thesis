from dto import NodeStatePower, NodeStateCost

from .cost_calculator import CostCalculator

class Generator:

  def __init__(self, generatorId, initialOutput, costProfile):
    self.generatorId = generatorId
    self.output = initialOutput # output => P_Gi (or Z in secondary control)
    self.costProfile = costProfile # Parameters used to estimate the cost to generate current output

  def getId(self) -> str:
    return self.generatorId

  def setOutput(self, newOutput) -> None:
    self.output = newOutput

  def getOutput(self) -> float:
    return self.output

  def toNodeStatePower(self) -> NodeStatePower:
    return NodeStatePower(id_=self.generatorId, power=self.getOutput())

  def toNodeStateCost(self) -> NodeStateCost:
    cost = CostCalculator.calculateCost(self.getOutput(), self.costProfile)
    return NodeStateCost(id_=self.generatorId, cost=cost)
