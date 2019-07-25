from dto import ElectricalConstants

class AreaDynamics:

  def __init__(self, electricalConstants: ElectricalConstants):
    self.inertia = electricalConstants.inertia            # inertia => M
    self.dampening = electricalConstants.dampening        # dampening => D
    self.timeConstant = electricalConstants.timeConstant  # timeConstant => Tg
    self.droop = electricalConstants.droop                # droop => Rd
    self.nominalFrequency = electricalConstants.nominalFrequency # nominalFrequency = (f_nom or f_0)

  def getDeltaFrequency(self, frequency):
    return self.nominalFrequency - frequency

  def calculatePowerGeneratedNew(self, zg, powGeneratedOld, deltaFreq): # zg = total control action (sum of generators Z)
    rd = self.droop
    tg = self.timeConstant

    powerGeneratedNew = powGeneratedOld + (- powGeneratedOld + zg - deltaFreq/rd)/tg

    return powerGeneratedNew

  def calculateFrequencyNew(self, powerGeneratedNew, totalLoad, frequencyOld):
    deltaFreqOld = self.getDeltaFrequency(frequencyOld)
    m = self.inertia
    d = self.dampening
    deltaFreqNew = deltaFreqOld + (powerGeneratedNew - totalLoad - d*deltaFreqOld)/m

    newFrequency = self.nominalFrequency + deltaFreqNew
