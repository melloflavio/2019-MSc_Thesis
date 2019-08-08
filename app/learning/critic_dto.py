from typing import List
from dataclasses import dataclass

@dataclass
class CriticEstimateInput:
  state: List[List[any]]
  actionActor: float
  actionsOthers: List[float]
  ltsmInternalState: tuple
  batchSize: int
  traceLength: int
