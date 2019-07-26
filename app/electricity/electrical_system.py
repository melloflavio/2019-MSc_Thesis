#!/usr/bin/env python3

from dto import ElectricalConstants, SystemHistory

from .area_dynamics import AreaDynamics

class ElectricalSystem:
  def __init__(self, electricalConstants: ElectricalConstants, loads, generators):
    self.areaDynamics = AreaDynamics(electricalConstants)
    self.loads = loads
    self.generators = generators
    self.history = SystemHistory(
      totalPower = [],
      frequency =  [],
      generators = { gen.getId(): [gen.getOutput()] for gen in self.generators },
      loads = { load.getId(): [load.getLoad()] for load in self.loads },
    )

  def getHistory(self):
    return self.history
