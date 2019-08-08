from typing import NamedTuple, Dict
from singleton_decorator import singleton

class NeuralNetworkShape(NamedTuple):
  layer_00_ltsm:   int # Initial LTSM layer
  layer_01_mlp_01: int # MLP Layer 1
  layer_02_mlp_02: int # MLP Layer 2
  layer_03_mlp_03: int # MLP Layer 3
  layer_04_mlp_04: int # MLP Layer 4 (output layer)

@singleton
class LearningParams(NamedTuple):
  gamma: float    = 0.9   # Gamma (Discount)
  tau: float      = 0.001 # Tau
  epsilon: float  = 0.99  # Epsilon
  numEpisodes: int   = 50000 # Number of learning episodes to run
  maxSteps: int      = 100   # Number of steps per learning episode
  traceSize: int  = 8     # Number of steps each sampled episode should contain
  batchSize: int  = 4     # Number of episodes sampled from experience buffer
  nnShape: NeuralNetworkShape = NeuralNetworkShape(
      layer_00_ltsm = 100,    # Initial LTSM layer
      layer_01_mlp_01 = 1000, # MLP Layer 1
      layer_02_mlp_02 = 100,  # MLP Layer 2
      layer_03_mlp_03 = 50,   # MLP Layer 3
      layer_04_mlp_04 = 1,   # MLP Layer 4 (output layer)
  )
