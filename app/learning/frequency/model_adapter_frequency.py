from ..model_adapter import ModelAdapter

from .nn_extensions_frequency import ActorFrequency, CriticFrequency

class ModelAdapterFrequency(ModelAdapter):

  @property
  def SCOPE_PREFIX(self):
    return 'freq'

  @property
  def Actor(self):
    return ActorFrequency

  @property
  def Critic(self):
    return CriticFrequency

  @classmethod
  def _defaultRewardFunction(deltaFreq):
    baseComponent = 2**(-(deltaFreq**2)/100)
    peakComponent = 2**(-(deltaFreq**2)/2)
    earnedReward = (baseComponent + 9*peakComponent)/10

    return earnedReward, {'base': baseComponent, 'peak': peakComponent, 'total':earnedReward}

  def shouldStopEarly(self, elecSystem):
    deltaFreq = elecSystem.getCurrentDeltaF()
    shouldStop = abs(deltaFreq) > 50
    return shouldStop

  def storeInitialState(self, elecSystem, allAgents):
    """Stores initial state. May be used for reward and future state observations"""
    pass

  ## TODO enforce typing
  def observeStates(self, elecSystem, allAgents):
    deltaFreq = elecSystem.getCurrentDeltaF()
    allStates = {agent.getId(): {
      'deltaFreq':deltaFreq
      } for agent in allAgents}
    return allStates

  def storePreActionStateReward(self, elecSystem):
    '''Stores state values which may be used later to calcluate reward. Called before system executes actions.'''
    pass

  def calculateReward(self, elecSystem):
    deltaFreq = elecSystem.getCurrentDeltaF()
    earnedReward, rewardComponents = self._rewardFn(deltaFreq=deltaFreq)
    return earnedReward, rewardComponents
