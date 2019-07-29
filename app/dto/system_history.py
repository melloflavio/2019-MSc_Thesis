from typing import Dict, List, NamedTuple
from singleton_decorator import singleton

@singleton
class SystemHistory(NamedTuple):
  totalPower: List[float] = []
  frequency: List[float] = []
  generators: Dict[str, List[float]] = {}
  loads: Dict[str, List[float]] = {}
