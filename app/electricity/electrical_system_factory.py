import random
import pydash as _

from dto import ElectricalSystemSpecs, ElectricalConstants

from .load import Load
from .generator import Generator
from .electrical_system import ElectricalSystem

class ElectricalSystemFactory:

  @staticmethod
  def create(specs: ElectricalSystemSpecs):
    allLoads = [Load(l.id_, ElectricalSystemFactory.applyNoise(l.basePower, l.noiseLevel))
                for l in specs.loads]
    allGenerators = [
        Generator(
            id_=g.id_,
            initialOutput=ElectricalSystemFactory.applyNoise(g.basePower, g.noiseLevel),
            costProfile=g.costProfile,
            minPower=g.minPower,
            maxPower=g.maxPower,
        ) for g in specs.generators]
    initialFrequency = ElectricalConstants().nominalFrequency

    system = ElectricalSystem(initialFrequency, allLoads, allGenerators)

    return system

  @staticmethod
  def applyNoise(basePower, noiseLevel):
    """Applies uniform noiseLevel (in %) to the base value"""
    return basePower + random.uniform(-basePower*noiseLevel, basePower*noiseLevel)
