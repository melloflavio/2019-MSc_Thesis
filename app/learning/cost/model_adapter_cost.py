from ..model_adapter import ModelAdapter

from .nn_extensions_cost import ActorCost, CriticCost

class ModelAdapterCost(ModelAdapter):
  _rewardTotalOutputOrigin = None
  _initialOutput = None

  @property
  def SCOPE_PREFIX(self):
    return 'cost'

  @property
  def Actor(self):
    return ActorCost

  @property
  def Critic(self):
    return CriticCost

  @classmethod
  def _defaultRewardFunction(totalCost, outputDifferential):
    scaledCost = totalCost/(10000*100.0) # Scale down cost to levels near the ones found in output differential (e.g. 10 */ 10)
    costComponent = 2**(-1*(totalCost**2)*200)
    outputComponent = 2**(-1*(outputDifferential**2)*500)
    earnedReward = costComponent*outputComponent

    return earnedReward, {'cost': costComponent, 'output': outputComponent, 'total':earnedReward}

  def shouldStopEarly(self, elecSystem):
    costDifferential = elecSystem.getCostOptimalDiferential()
    shouldStop = abs(costDifferential) > 50
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
    generatorsOutputs = elecSystem.getGeneratorsOutputs()
    totalOutput = sum(generatorsOutputs.values())
    self._rewardTotalOutputOrigin = totalOutput

  def calculateReward(self, elecSystem):
    totalCost = elecSystem.getTotalCost()

    # If initial output is tracked, the reward is for training/testing, thus compared against the initial output
    # Otherwise, it is for execution and thus compared against the output prior to taking the step
    totalOutputOrigin = self._initialOutput

    generatorsOutputs = elecSystem.getGeneratorsOutputs()
    totalOutputDestination = sum(generatorsOutputs.values())
    outputDifferential = (totalOutputDestination - totalOutputOrigin)/totalOutputOrigin

    earnedReward, rewardComponents = self._rewardFn(totalCost=totalCost, outputDifferential=outputDifferential)
    return earnedReward, rewardComponents
