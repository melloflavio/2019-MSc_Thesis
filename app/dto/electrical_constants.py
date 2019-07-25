from typing import NamedTuple

class ElectricalConstants(NamedTuple):
  inertia: float            # inertia => M
  dampening: float        # dampening => D
  timeConstant: float  # timeConstant => Tg
  droop: float                # droop => Rd
  nominalFrequency: float # nominalFrequency = (f_nom or f_0)
