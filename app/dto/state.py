from typing import List, NamedTuple

class NodeStatePower(NamedTuple):
  id_: str
  power: float

class State(NamedTuple):
  totalPower: float
  frequency: float
  loads: List[NodeStatePower]
  generators: List[NodeStatePower]
