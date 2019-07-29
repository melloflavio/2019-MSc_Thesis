from typing import List, NamedTuple

class NodeState(NamedTuple):
  id_: str
  power: float

class State(NamedTuple):
  totalPower: float
  frequency: float
  loads: List[NodeState]
  generators: List[NodeState]
