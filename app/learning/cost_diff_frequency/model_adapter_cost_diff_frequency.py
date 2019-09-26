from ..model_adapter import ModelAdapter

from .nn_extensions_cost_diff_frequency import ActorCostDiffFrequency, CriticCostDiffFrequency

class ModelAdapterCostDiffFrequency(ModelAdapter):

  @property
  def SCOPE_PREFIX(self):
    return 'cost_diff_freq'

  @property
  def Actor(self):
    return ActorCostDiffFrequency

  @property
  def Critic(self):
    return CriticCostDiffFrequency

  @classmethod
  def _defaultRewardFunction(deltaFreq, costDifferential):
    costComponent = 2**(-1*(costDifferential**2)/4)
    freqComponent = 2**(-1*(deltaFreq**2)/2)
    earnedReward = costComponent*freqComponent

    return earnedReward, {'cost': costComponent, 'freq': freqComponent, 'total':earnedReward}

  def shouldStopEarly(self, elecSystem):
    costDifferential = elecSystem.getCostOptimalDiferential()
    shouldStop = abs(costDifferential) > 2
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
    costDifferential = elecSystem.getCostOptimalDiferential()
    earnedReward, rewardComponents = self._rewardFn(deltaFreq=deltaFreq, costDifferential=costDifferential)
    return earnedReward, rewardComponents
