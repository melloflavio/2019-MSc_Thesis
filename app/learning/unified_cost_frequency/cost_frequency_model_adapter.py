from .cost_frequency_actor import ActorCostFrequency
from .cost_frequency_critic import CriticCostFrequency

# def defaultRewardFunction(deltaFreq, totalCost):
#     totalCost = totalCost/(10000.0) # Scale down cost to levels near the ones found in output differential (e.g. 10 */ 10)
#     costComponent = 2**(-1*(totalCost**2)/50)
#     freqComponent = 2**(-1*(deltaFreq**2)/2)
#     earnedReward = 10*costComponent*freqComponent

#     return earnedReward, {'cost': costComponent, 'freq': freqComponent, 'total':earnedReward}

def _defaultRewardFunction(deltaFreq, totalCost):
    totalCost = totalCost/(10000.0) # Scale down cost to levels near the ones found in output differential (e.g. 10 */ 10)
    costComponent = 2**(-1*(totalCost**2)/50)
    freqComponent = 2**(-1*(deltaFreq**2)/2)
    earnedReward = 10*costComponent*freqComponent

    return earnedReward, {'cost': costComponent, 'freq': freqComponent, 'total':earnedReward}


class ModelAdapterCostFrequency():

  SCOPE_PREFIX = 'cost'

  Actor = ActorCostFrequency
  Critic = CriticCostFrequency

  def __init__(self, rewardFn=_defaultRewardFunction):
    self._rewardFn = rewardFn

  def shouldStopEarly(self, elecSystem):
    costDifferential = elecSystem.getCostOptimalDiferential()
    shouldStop = abs(costDifferential) > 50
    return shouldStop

  ## TODO enforce typing
  def observeStates(self, elecSystem):
    deltaFreqOriginal = elecSystem.getCurrentDeltaF()
    generatorsOutputsOrigin = elecSystem.getGeneratorsOutputs()
    totalOutputOrigin = sum(generatorsOutputsOrigin.values())
    allStatesOrigin = {actorId: {'genOutput': output, 'totalOutput':totalOutputOrigin, 'deltaFreq':deltaFreqOriginal} for actorId, output in generatorsOutputsOrigin.items()}
    return allStatesOrigin

  def calculateReward(self, elecSystem):
    deltaFreq = elecSystem.getCurrentDeltaF()
    totalCost = elecSystem.getTotalCost()
    earnedReward, rewardComponents = self._rewardFn(deltaFreq=deltaFreq, totalCost=totalCost)
    return earnedReward, rewardComponents
