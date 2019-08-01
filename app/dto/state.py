from typing import List, NamedTuple

class NodeStatePower(NamedTuple):
  id_: str
  power: float

class NodeStateCost(NamedTuple):
  id_: str
  cost: float

class State(NamedTuple):
  totalPower: float
  totalLoad: float
  frequency: float
  loads: List[NodeStatePower]
  generators: List[NodeStatePower]
  actualCost: List[NodeStateCost]
  optimalCost: List[NodeStateCost]
  totalCost: List[NodeStateCost]
