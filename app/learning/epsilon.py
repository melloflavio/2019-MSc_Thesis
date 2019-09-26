from dto import EpsilonSpecs

# Instead of manually setting the decay rates, we establish the threshold value and at which
# point in the experiment this value should be reached. This makes for a less brittle experimenting
# environment as changes in experiment length (numEpisodes/maxSteps) automatically reflect on the
# epsilon decay rates
# Keep in mind that the decay change is not exactly as specified since early episode termination
# means there will be less steps taken in total. Thus the de facto change will occur at a later
# point (eg. specified to change at 60% of learning, happens at 68%)
class Epsilon():
  def __init__(self, specs: EpsilonSpecs, numEpisodes: int, stepsPerEpisode: int):
    totalSteps = numEpisodes*stepsPerEpisode
    stepsUntilThreshold = totalSteps*specs.thresholdProgress
    stepsAfterThreshold = totalSteps - stepsUntilThreshold

    self.decayRate_1 = specs.thresholdValue**(1.0/stepsUntilThreshold) # **(1/n) => nth root
    self.decayRate_2 = specs.finalValue**(1.0/stepsAfterThreshold) # **(1/n) => nth root

    self.epsilonVal = self.decayRate_1 # Use decay rate as initial value
    self.threshold = specs.thresholdValue

  @property
  def value(self):
    return self.epsilonVal

  def decay(self):
    rate = self.decayRate_1 if self.epsilonVal > self.threshold else self.decayRate_2
    self.epsilonVal *= rate
