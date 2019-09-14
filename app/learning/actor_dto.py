from typing import List
from dataclasses import dataclass

@dataclass
class ActionInput:
  actorInput: List[List[any]]
  ltsmInternalState: tuple
  batchSize: int
  traceLength: int

@dataclass
class ActionOutput:
  action: float
  nextState: tuple

@dataclass
class ActorUpdateInput:
  state: List[List[any]]
  gradients: any
  batchSize: int
  traceLength: int
