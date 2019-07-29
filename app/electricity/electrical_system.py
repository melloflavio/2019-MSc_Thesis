from dto import State, SystemHistory

class ElectricalSystem:
  def __init__(self, initialFrequency, loads, generators):
    self.loads = loads
    self.generators = generators

    initialPower = sum([gen.getOutput() for gen in self.generators])
    SystemHistory().pushState(State(
        totalPower=initialPower,
        frequency=initialFrequency,
        loads=[l.toNodeState() for l in self.loads],
        generators=[g.toNodeState() for g in self.generators]),
        )
