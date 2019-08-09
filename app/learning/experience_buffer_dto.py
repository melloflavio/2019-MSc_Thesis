from typing import Dict, List
from dataclasses import dataclass

@dataclass
class XpMiniBatch:
  originalStates: List[List[float]]
  destinationStates: List[List[float]]
  groupedActions: Dict[str, List[List[float]]]
  rewards: List[List[float]]
