from typing import List, NamedTuple

class LoadSpecs(NamedTuple):
  id_: str
  basePower: float

class GeneratorSpecs(NamedTuple):
  id_: str
  basePower: float
  costProfile: any
  minPower: float
  maxPower: float

class ElectricalSystemSpecs(NamedTuple):
  loads: List[LoadSpecs]
  generators: List[GeneratorSpecs]
  noiseLevel: float # % Noise applied to evey base power (10% => .1)
