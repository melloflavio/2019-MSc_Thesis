from dto import ElectricalConstants

class AreaDynamics:

  def getDeltaFrequency(frequency):
    return ElectricalConstants().nominalFrequency - frequency

  def calculatePowerGeneratedNew(zg, powGeneratedOld, deltaFreq): # zg = total control action (sum of generators Z)
    rd = ElectricalConstants().droop
    tg = ElectricalConstants().timeConstant

    powerGeneratedNew = powGeneratedOld + (- powGeneratedOld + zg - deltaFreq/rd)/tg

    return powerGeneratedNew

  def calculateFrequencyNew(powerGeneratedNew, totalLoad, frequencyOld):
    deltaFreqOld = AreaDynamics.getDeltaFrequency(frequencyOld)
    m = ElectricalConstants().inertia
    d = ElectricalConstants().dampening
    deltaFreqNew = deltaFreqOld + (powerGeneratedNew - totalLoad - d*deltaFreqOld)/m

    newFrequency = ElectricalConstants().nominalFrequency + deltaFreqNew
