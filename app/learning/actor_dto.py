from typing import List
from dataclasses import dataclass

@dataclass
class ActionInput:
  actorInput: List[List[any]]
  actorState: tuple
  batchSize: int
  traceLength: int

@dataclass
class ActionOutput:
  action: float
  nextState: tuple
