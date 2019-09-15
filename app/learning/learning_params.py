from dataclasses import dataclass
from dataclasses_json import dataclass_json
from singleton_decorator import singleton

from dto import ElectricalSystemSpecs, EpsilonSpecs

@dataclass_json
@dataclass
class NeuralNetworkShape:
  layer_00_ltsm:   int # Initial LTSM layer
  layer_01_mlp_01: int # MLP Layer 1
  layer_02_mlp_02: int # MLP Layer 2
  layer_03_mlp_03: int # MLP Layer 3
  layer_04_mlp_04: int # MLP Layer 4 (output layer)

@singleton
@dataclass_json
@dataclass
class LearningParams:
  electricalSystemSpecs: ElectricalSystemSpecs
  gamma: float    = 0.9   # Gamma (Discount)
  tau: float      = 0.001 # Tau
  epsilonSpecs: EpsilonSpecs = EpsilonSpecs( # Epsilon explore/exploit control
      thresholdProgress = 0.6, # % of steps where decay change should happen
      thresholdValue = 0.5, # Value at which decay change would happen
      finalValue = 0.0001, # Value at the end of the experiment
  )
  numEpisodes: int   = 1000 # Number of learning episodes to run
  maxSteps: int      = 200   # Number of steps per learning episode
  bufferSize: int = 100  # Experience Buffer Size
  traceLength: int  = 8     # Number of steps each sampled episode should contain
  batchSize: int  = 4     # Number of episodes sampled from experience buffer
  nnShape: NeuralNetworkShape = NeuralNetworkShape(
      layer_00_ltsm = 100,    # Initial LTSM layer
      layer_01_mlp_01 = 1000, # MLP Layer 1
      layer_02_mlp_02 = 100,  # MLP Layer 2
      layer_03_mlp_03 = 50,   # MLP Layer 3
      layer_04_mlp_04 = 1,   # MLP Layer 4 (output layer)
  )
  modelName: str = 'DefaultModel'
