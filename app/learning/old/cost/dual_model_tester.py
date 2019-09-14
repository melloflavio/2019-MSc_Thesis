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

class DualModelTester():
  def __init__(self):
    self._tfSessions = {
        FREQUENCY: None,
        COST: None,
    }

    self._allAgents = {
        FREQUENCY: None,
        COST: None,
    }

    self._allRewards = {
        FREQUENCY: [],
        COST: [],
        'cost_components': []
    }

  def loadTfModels(self, modelNameFreq: str, modelNameCost: str, generatorIds):
    # Declare both models TF graphs and sessions
    tfGraphs = {
      FREQUENCY: tf.Graph(),
      COST: tf.Graph(),
    }
    self._tfSessions[FREQUENCY] = tf.compat.v1.Session(graph=tfGraphs[FREQUENCY])
    self._tfSessions[COST] = tf.compat.v1.Session(graph=tfGraphs[COST])

    tfSavers = {}
    # Recreate agents within each graph
    with tfGraphs[FREQUENCY].as_default():
      self._allAgents[FREQUENCY]: List[Agent] = [Agent(id_) for id_ in generatorIds]
      tfSavers[FREQUENCY] = tf.compat.v1.train.Saver()
    with tfGraphs[COST].as_default():
      self._allAgents[COST]: List[CostAgent] = [CostAgent(id_) for id_ in generatorIds]
      tfSavers[COST] = tf.compat.v1.train.Saver()

    # Restore TF sessions

    modelPathFreq = getPathForModel(modelNameFreq)
    tfSavers[FREQUENCY].restore(self._tfSessions[FREQUENCY], modelPathFreq)
    modelPathCost = getPathForModel(modelNameCost)
    tfSavers[COST].restore(self._tfSessions[COST], modelPathCost)

  def testAgents(self, electricalSystemSpecs: ElectricalSystemSpecs, modelNameFreq: str, modelNameCost: str, rewardFnCost=costRewardFunction, stepsToTest: int = 500, frequencyWeight=0.7):

    # Recreate simulation environment
    elecSystem = ElectricalSystemFactory.create(electricalSystemSpecs)

    self.loadTfModels(
      modelNameFreq=modelNameFreq,
      modelNameCost=modelNameCost,
      generatorIds=[generator.id_ for generator in electricalSystemSpecs.generators],
    )

    for stepIdx in range(stepsToTest):
      # For cost reward
      generatorsOutputsOrigin = elecSystem.getGeneratorsOutputs()
      totalOutputOrigin = sum(generatorsOutputsOrigin.values())

      allActionsFreq = self.getActionsFreq(elecSystem)
      allActionsCost = self.getActionsCost(elecSystem)

      costWeight = 1 - frequencyWeight

      actionsCombined = [frequencyWeight*actionFreq + costWeight*actionCost for (actionFreq, actionCost) in zip(allActionsFreq, allActionsCost)]

      # Execute agents'actions (i.e. update the generators' power output)
      agentIds = [agent.getId() for agent in self._allAgents[FREQUENCY]] # Both agent collections have the same ids...
      generatorUpdates = [NodePowerUpdate(
          id_=agentId,
          deltaPower=action
        ) for (agentId, action) in zip(agentIds, actionsCombined)]
      elecSystem.updateGenerators(generatorUpdates)

      # Calculate earned rewards per objective
      # Frequency
      deltaFreqDestination = elecSystem.getCurrentDeltaF()
      freqReward = 2**(10-abs(deltaFreqDestination))
      self._allRewards[FREQUENCY].append(freqReward) # TODO Calculate reward according to a given strategy

      # Cost
      generatorsOutputsDestination = elecSystem.getGeneratorsOutputs()
      totalOutputDestination = sum(generatorsOutputsDestination.values())
      totalCost = elecSystem.getTotalCost()
      rewardCost, rewardComponentsCost = rewardFnCost(totalOutputTarget=totalOutputOrigin, totalOutputDestination=totalOutputDestination, totalCost=totalCost)
      self._allRewards[COST].append(rewardCost)
      self._allRewards['cost_components'].append(rewardComponentsCost)

    # Close tf sessions after using them
    for tfSession in self._tfSessions.values():
      tfSession.close()

    return elecSystem, self._allRewards

  def getActionsCost(self, elecSystem):
    generatorsOutputsOrigin = elecSystem.getGeneratorsOutputs()
    totalOutput = sum(generatorsOutputsOrigin.values())

    allStatesOrigin = {actorId: {'genOutput': output, 'totalOutput':totalOutput} for actorId, output in generatorsOutputsOrigin.items()}
    actionsCost = [agent.runActorAction(self._tfSessions[COST], allStatesOrigin.get(agent.getId())) for agent in self._allAgents[COST]]
    actionsCost = [action[0, 0] for action in actionsCost]

    return actionsCost

  def getActionsFreq(self, elecSystem):
    deltaFreqOriginal = elecSystem.getCurrentDeltaF()
    actionsFreq = [agent.runActorAction(self._tfSessions[FREQUENCY], deltaFreqOriginal) for agent in self._allAgents[FREQUENCY]]
    actionsFreq = [action[0, 0] for action in actionsFreq]

    return actionsFreq
