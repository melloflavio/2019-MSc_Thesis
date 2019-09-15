from ..model_adapter import ModelAdapter

from .nn_extensions_cost import ActorCost, CriticCost

class ModelAdapterCost(ModelAdapter):
  _totalOutputOrigin = None

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

  ## TODO enforce typing
  def observeStates(self, elecSystem, allAgents):
    generatorsOutputs = elecSystem.getGeneratorsOutputs()
    totalOutput = sum(generatorsOutputs.values())
    allStates = {actorId: {
      'genOutput': output,
      'totalOutput':totalOutput,
      } for actorId, output in generatorsOutputs.items()}
    return allStates

  def storePreactionStateReward(self, elecSystem):
    '''Stores state values which may be used later to calcluate reward. Called before system executes actions.'''
    generatorsOutputs = elecSystem.getGeneratorsOutputs()
    totalOutput = sum(generatorsOutputs.values())
    self._totalOutputOrigin = totalOutput

  def calculateReward(self, elecSystem):
    totalCost = elecSystem.getTotalCost()

    generatorsOutputs = elecSystem.getGeneratorsOutputs()
    totalOutputDestination = sum(generatorsOutputs.values())
    outputDifferential = (totalOutputDestination - self._totalOutputOrigin)/self._totalOutputOrigin

    earnedReward, rewardComponents = self._rewardFn(totalCost=totalCost, outputDifferential=outputDifferential)
    return earnedReward, rewardComponents
