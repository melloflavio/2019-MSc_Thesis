from dto import SystemHistory

class ElectricalSystem:
  def __init__(self, initialFrequency, loads, generators):
    self.loads = loads
    self.generators = generators

    initialPower = sum([gen.getOutput() for gen in self.generators])
    SystemHistory().pushState(initialPower, initialFrequency, self.loads, self.generators)
