from typing import List
from dataclasses import dataclass

@dataclass
class CriticEstimateInput:
  state: List[List[any]]
  actionActor: float
  actionsOthers: List[float]
  ltsmInState: tuple
  batchSize: int
  traceLength: int

#             self.state: s_prime,
#             self.action: a_target_1,
#             self.actionOthers: a_target_2,
#             self.trainLength:trace,
#             self.batchSize:batch,
#             self.stateIn:state_train
