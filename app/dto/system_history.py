from typing import Dict, List, NamedTuple

class SystemHistory(NamedTuple):
  totalPower: List[float]
  frequency: List[float]
  generators: Dict[str, List[float]]
  loads: Dict[str, List[float]]
