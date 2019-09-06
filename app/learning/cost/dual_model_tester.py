from typing import List
import tensorflow as tf

from electricity import ElectricalSystemFactory
from dto import ElectricalSystemSpecs, NodePowerUpdate
from models import getPathForModel

from .cost_reward import costRewardFunction
from .cost_learning_agent import CostAgent
from ..learning_agent import Agent

FREQUENCY = 'frequency'
COST = 'cost'

_tfSessions = {
    FREQUENCY: None,
    COST: None,
  }

_allAgents = {
  FREQUENCY: None,
  COST: None,
}

_allRewards = {
  FREQUENCY: [],
  COST: [],
  'cost_components': []
}

class DualModelTester():

  @staticmethod
  def loadTfModels(modelNameFreq: str, modelNameCost: str, generatorIds):
    # Declare both models TF graphs and sessions
    tfGraphs = {
      FREQUENCY: tf.Graph(),
      COST: tf.Graph(),
    }
    _tfSessions[FREQUENCY] = tf.compat.v1.Session(graph=tfGraphs[FREQUENCY])
    _tfSessions[COST] = tf.compat.v1.Session(graph=tfGraphs[COST])

    # Recreate agents within each graph
    with tfGraphs[FREQUENCY].as_default():
      _allAgents[FREQUENCY]: List[Agent] = [Agent(id_) for id_ in generatorIds]
    with tfGraphs[COST].as_default():
      _allAgents[COST]: List[CostAgent] = [CostAgent(id_) for id_ in generatorIds]

    # Restore TF sessions
    tfSaver = tf.compat.v1.train.Saver() # Saver obj used to restore session
    modelPathFreq = getPathForModel(modelNameFreq)
    tfSaver.restore(_tfSessions[FREQUENCY], modelPathFreq)
    modelPathCost = getPathForModel(modelNameCost)
    tfSaver.restore(_tfSessions[COST], modelPathCost)

  @staticmethod
  def testAgents(electricalSystemSpecs: ElectricalSystemSpecs, modelNameFreq: str, modelNameCost: str, rewardFnCost=costRewardFunction, stepsToTest: int = 500, frequencyWeight=0.7):

    # Recreate simulation environment
    elecSystem = ElectricalSystemFactory.create(electricalSystemSpecs)
    _initTotalZ = sum(elecSystem.getGeneratorsOutputs().values())

    DualModelTester.loadTfModels(
      modelNameFreq=modelNameFreq,
      modelNameCost=modelNameCost,
      generatorIds=[generator.id_ for generator in electricalSystemSpecs.generators],
    )

    for stepIdx in range(stepsToTest):
      # For cost reward
      generatorsOutputsOrigin = elecSystem.getGeneratorsOutputs()
      totalOutputOrigin = sum(generatorsOutputsOrigin.values())

      allActionsFreq = DualModelTester.getActionsFreq(elecSystem)
      allActionsCost = DualModelTester.getActionsCost(elecSystem)

      costWeight = 1 - frequencyWeight

      actionsCombined = [frequencyWeight*actionFreq + costWeight*actionCost for (actionFreq, actionCost) in zip(allActionsFreq, allActionsCost)]

      # Execute agents'actions (i.e. update the generators' power output)
      agentIds = [agent.getId() for agent in _allAgents[FREQUENCY]] # Both agent collections have the same ids...
      generatorUpdates = [NodePowerUpdate(
          id_=agentId,
          deltaPower=action
        ) for (agentId, action) in zip(agentIds, actionsCombined)]
      elecSystem.updateGenerators(generatorUpdates)

      # Calculate earned rewards per objective
      # Frequency
      deltaFreqDestination = elecSystem.getCurrentDeltaF()
      _allRewards[FREQUENCY].append(2**(10-abs(deltaFreqDestination))) # TODO Calculate reward according to a given strategy

      # Cost
      generatorsOutputsDestination = elecSystem.getGeneratorsOutputs()
      totalOutputDestination = sum(generatorsOutputsDestination.values())
      totalCost = elecSystem.getTotalCost()
      rewardCost, rewardComponentsCost = rewardFnCost(totalOutputTarget=totalOutputOrigin,  totalOutputDestination=totalOutputDestination, totalCost=totalCost)
      _allRewards[COST].append(rewardCost)
      _allRewards['cost_components'].append(rewardComponentsCost)

    # Close tf sessions after using them
    for tfSession in _tfSessions.values():
      tfSession.close()

    return elecSystem, _allRewards

  @staticmethod
  def getActionsCost(elecSystem):
    generatorsOutputsOrigin = elecSystem.getGeneratorsOutputs()
    totalOutput = sum(generatorsOutputsOrigin.values())

    allStatesOrigin = {actorId: {'genOutput': output, 'totalOutput':totalOutput} for actorId, output in generatorsOutputsOrigin.items()}
    actionsCost = [agent.runActorAction(_tfSessions[COST], allStatesOrigin.get(agent.getId())) for agent in _allAgents[COST]]
    actionsCost = [action[0, 0] for action in actionsCost]

    return actionsCost

  @staticmethod
  def getActionsFreq(elecSystem):
    deltaFreqOriginal = elecSystem.getCurrentDeltaF()
    actionsFreq = [agent.runActorAction(_tfSessions[FREQUENCY], deltaFreqOriginal) for agent in _allAgents[FREQUENCY]]
    actionsFreq = [action[0, 0] for action in actionsFreq]

    return actionsFreq
