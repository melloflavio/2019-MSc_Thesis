from ..model_adapter import ModelAdapter

from .nn_extensions_cost_single import ActorCostSingle, CriticCostSingle

class ModelAdapterCostSingle(ModelAdapter):
  _initialOutput = None

  @property
  def SCOPE_PREFIX(self):
    return 'cost_single'

  @property
  def Actor(self):
    return ActorCostSingle

  @property
  def Critic(self):
    return CriticCostSingle

  @classmethod
  def _defaultRewardFunction(outputDifferentialFromOpt):
    baseComponent = 2**(-(outputDifferentialFromOpt**2)/100)
    peakComponent = 2**(-(outputDifferentialFromOpt**2)/2)
    earnedReward = (baseComponent + 9*peakComponent)/10

    return earnedReward, {'base': baseComponent, 'peak': peakComponent, 'total':earnedReward}

  def shouldStopEarly(self, elecSystem):
    outputDifferential = elecSystem.getOptimalDifferentialFromInitialState()
    shouldStop = abs(outputDifferential) > 50
    print(f'OutputDiff:{outputDifferential}')
    return shouldStop

  def storeInitialState(self, elecSystem, allAgents):
    """Stores initial state. May be used for reward and future state observations"""
    generatorsOutputs = elecSystem.getGeneratorsOutputs()
    totalOutput = sum(generatorsOutputs.values())
    self._initialOutput = totalOutput

  ## TODO enforce typing
  def observeStates(self, elecSystem, allAgents):
    generatorsOutputs = elecSystem.getGeneratorsOutputs()
    totalOutput = sum(generatorsOutputs.values())
    allStates = {actorId: {
      'genOutput': output,
      'totalOutput':self._initialOutput,
      } for actorId, output in generatorsOutputs.items()}
    return allStates

  def storePreActionStateReward(self, elecSystem):
    '''Stores state values which may be used later to calcluate reward. Called before system executes actions.'''
    pass

  def calculateReward(self, elecSystem):
    # totalCost = elecSystem.getTotalCost()

    # # If initial output is tracked, the reward is for training/testing, thus compared against the initial output
    # # Otherwise, it is for execution and thus compared against the output prior to taking the step
    # totalOutputOrigin = self._initialOutput

    # generatorsOutputs = elecSystem.getGeneratorsOutputs()
    # totalOutputDestination = sum(generatorsOutputs.values())
    # outputDifferential = (totalOutputDestination - totalOutputOrigin)/totalOutputOrigin

    # earnedReward, rewardComponents = self._rewardFn(totalCost=totalCost, outputDifferential=outputDifferential)
    # return earnedReward, rewardComponents
    outputDiff = elecSystem.getOptimalDifferentialFromInitialState()
    earnedReward, rewardComponents = self._rewardFn(outputDifferentialFromOpt=outputDiff)
    return earnedReward, rewardComponents
