#!/usr/bin/env python3

from dto import ElectricalConstants, SystemHistory

from .area_dynamics import AreaDynamics

class ElectricalSystem:
  def __init__(self, electricalConstants: ElectricalConstants, loads, generators):
    self.areaDynamics = AreaDynamics(electricalConstants)
    self.loads = loads
    self.generators = generators
