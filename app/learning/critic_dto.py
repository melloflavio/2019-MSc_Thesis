from typing import List
from dataclasses import dataclass

@dataclass
class CriticEstimateInput:
  state: List[List[float]]
  actionActor: float
  actionsOthers: List[float]
  ltsmInternalState: tuple
  batchSize: int
  traceLength: int

@dataclass
class CriticUpdateInput:
  state: List[List[float]]
  actionActor: float
  actionsOthers: List[float]
  targetQs: List[List[float]]
  ltsmInternalState: tuple
  batchSize: int
  traceLength: int

@dataclass
class CriticGradientInput:
  state: List[List[float]]
  actionActor: float
  actionsOthers: List[float]
  ltsmInternalState: tuple
  batchSize: int
  traceLength: int
