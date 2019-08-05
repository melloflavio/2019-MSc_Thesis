from typing import NamedTuple
from singleton_decorator import singleton

@singleton
class LearningParams(NamedTuple):
  gamma: float    = 0.9   # Gamma (Discount)
  tau: float      = 0.001 # Tau
  epsilon: float  = 0.99  # Epsilon
  episodes: int   = 50000 # Number of learning episodes to run
  steps: int      = 100   # Number of steps per learning episode
  traceSize: int  = 8     # Number of steps each sampled episode should contain
  batchSize: int  = 4     # Number of episodes sampled from experience buffer
