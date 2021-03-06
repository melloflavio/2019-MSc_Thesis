from ..model_adapter import ModelAdapter

from .nn_extensions_cost_frequency import ActorCostFrequency, CriticCostFrequency

class ModelAdapterCostFrequency(ModelAdapter):

  @property
  def SCOPE_PREFIX(self):
    return 'cost_freq'

  @property
  def Actor(self):
    return ActorCostFrequency

  @property
  def Critic(self):
    return CriticCostFrequency

  @classmethod
  def _defaultRewardFunction(deltaFreq, totalCost):
    totalCost = totalCost/(10000.0) # Scale down cost to levels near the ones found in output differential (e.g. 10 */ 10)
    costComponent = 2**(-1*(totalCost**2)/50)
    freqComponent = 2**(-1*(deltaFreq**2)/2)
    earnedReward = 10*costComponent*freqComponent

    return earnedReward, {'cost': costComponent, 'freq': freqComponent, 'total':earnedReward}

  def shouldStopEarly(self, elecSystem):
    costDifferential = elecSystem.getCostOptimalDiferential()
    shouldStop = abs(costDifferential) > 50
    return shouldStop

  def storeInitialState(self, elecSystem, allAgents):
    """Stores initial state. May be used for reward and future state observations"""
    pass

  ## TODO enforce typing
  def observeStates(self, elecSystem, allAgents):
    deltaFreq = elecSystem.getCurrentDeltaF()
    generatorsOutputs = elecSystem.getGeneratorsOutputs()
    totalOutput = sum(generatorsOutputs.values())
    allStates = {actorId: {
      'genOutput': output,
      'totalOutput':totalOutput,
      'deltaFreq':deltaFreq
      } for actorId, output in generatorsOutputs.items()}
    return allStates

  def storePreActionStateReward(self, elecSystem):
    '''Stores state values which may be used later to calcluate reward. Called before system executes actions.'''
    pass

  def calculateReward(self, elecSystem):
    deltaFreq = elecSystem.getCurrentDeltaF()
    totalCost = elecSystem.getTotalCost()
    earnedReward, rewardComponents = self._rewardFn(deltaFreq=deltaFreq, totalCost=totalCost)
    return earnedReward, rewardComponents
