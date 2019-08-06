from typing import List, NamedTuple

class NodeStatePower(NamedTuple):
  id_: str
  power: float

class NodeStateCost(NamedTuple):
  id_: str
  cost: float

class ElectricalState(NamedTuple):
  totalPower: float
  totalLoad: float
  frequency: float
  loads: List[NodeStatePower]
  generators: List[NodeStatePower]
  actualCost: List[NodeStateCost]
  costOptimalCost: List[NodeStateCost]
  costOptimalPower: List[NodeStatePower]
  totalCost: List[NodeStateCost]

class NodePowerUpdate(NamedTuple):
  id_: str
  deltaPower: float
