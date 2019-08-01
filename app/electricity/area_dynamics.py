from dto import ElectricalConstants

class AreaDynamics:

  @staticmethod
  def getDeltaFrequency(frequency):
    return frequency - ElectricalConstants().nominalFrequency

  @staticmethod
  def calculatePowerGeneratedNew(zg, powGeneratedOld, frequencyOld): # zg = total control action (sum of generators Z)
    deltaFreq = AreaDynamics.getDeltaFrequency(frequencyOld)
    rd = ElectricalConstants().droop
    tg = ElectricalConstants().timeConstant

    powerGeneratedNew = powGeneratedOld + (- powGeneratedOld + zg - deltaFreq/rd)/tg

    return powerGeneratedNew

  @staticmethod
  def calculateFrequencyNew(powerGeneratedNew, totalLoad, frequencyOld):
    deltaFreqOld = AreaDynamics.getDeltaFrequency(frequencyOld)
    m = ElectricalConstants().inertia
    d = ElectricalConstants().dampening
    deltaFreqNew = deltaFreqOld + (powerGeneratedNew - totalLoad - d*deltaFreqOld)/m

    newFrequency = ElectricalConstants().nominalFrequency + deltaFreqNew
    return newFrequency
