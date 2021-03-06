from typing import List, NamedTuple

class LoadSpecs(NamedTuple):
  id_: str
  basePower: float
  noiseLevel: float = 0 # % Noise applied to evey base power (10% => .1)

class GeneratorSpecs(NamedTuple):
  id_: str
  basePower: float
  costProfile: any
  minPower: float
  maxPower: float
  noiseLevel: float = 0 # % Noise applied to evey base power (10% => .1)

class ElectricalSystemSpecs(NamedTuple):
  loads: List[LoadSpecs]
  generators: List[GeneratorSpecs]
  shouldTrackOptimalCost: bool = True # Optimal cost calculation is computationally costly, if not using data in training it speeds up to not track it
