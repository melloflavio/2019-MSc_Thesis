import math
from dto import NodeStatePower, NodeStateCost

from .cost_calculator import CostCalculator

class Generator:

  def __init__(self, id_, initialOutput, costProfile, minPower=0, maxPower=math.inf):
    self.id_ = id_
    self.output = initialOutput # output => P_Gi (or Z in secondary control)
    self.costProfile = costProfile # Parameters used to estimate the cost to generate current output
    self.minPower = minPower
    self.maxPower = maxPower

  def getId(self) -> str:
    return self.id_

  def updateOutput(self, deltaOutput) -> None:
    self.output += deltaOutput

  def getOutput(self) -> float:
    return self.output

  def getCostProfile(self):
    return self.costProfile

  def getMinPower(self):
    return self.minPower

  def getMaxPower(self):
    return self.maxPower

  def getCost(self):
    return CostCalculator.calculateCost(self.getOutput(), self.costProfile)

  def toNodeStatePower(self) -> NodeStatePower:
    return NodeStatePower(id_=self.id_, power=self.getOutput())

  def toNodeStateCost(self) -> NodeStateCost:
    return NodeStateCost(id_=self.id_, cost=self.getCost())
