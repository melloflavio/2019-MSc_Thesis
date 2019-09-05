from typing import NamedTuple
from singleton_decorator import singleton
from dataclasses import dataclass

@singleton
@dataclass
class ElectricalConstants:
  inertia: float          = 0.1         # inertia => M
  dampening: float        = 0.0160      # dampening => D
  timeConstant: float     = 30          # timeConstant => Tg
  droop: float            = 0.1         # droop => Rd
  nominalFrequency: float = 50          # nominalFrequency = (f_nom or f_0)
  basePower: float        = 100         # Convert between pu and MVA (MVA = basePower*pu)
