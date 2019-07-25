from dto import SystemHistory

from .area_dynamics import AreaDynamics

class System:
  def __init__(self, electricalConstants: ElectricalConstants, loads, generators):
    self.areaDynamics = AreaDynamics(electricalConstants)
    self.loads = loads
    self.generators = generators
    self.history: SystemHistory = SystemHistory(
      totalPower = [],
      frequency =  [],
      generators = { gen.getId(): [gen.getOutput()] for gen in self.generators },
      loads = { load.getId(): [load.getOutput()] for load in self.loads },
    )

  def getHistory(self):
    return self.history
