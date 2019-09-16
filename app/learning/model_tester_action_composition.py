from typing import List
import tensorflow as tf

from electricity import ElectricalSystemFactory
from dto import ElectricalSystemSpecs, NodePowerUpdate
from models import getPathForModel

# from .cost_reward import costRewardFunction
# from .cost_learning_agent import CostAgent
from .learning_agent import Agent

FREQUENCY = 'frequency'
COST = 'cost'

MODEL_TYPES = [FREQUENCY, COST]

class ModelTesterActionComposition():
  def __init__(self, modelAdapterFreq: ModelAdapter, modelAdapterCost: ModelAdapter):

    self._modelAdapters: Dict[str, ModelAdapter] = {
        FREQUENCY: modelAdapterFreq,
        COST: modelAdapterCost,
    }

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
    }

    self._allRewardComponents = {
        FREQUENCY: [],
        COST: [],
    }

  def loadTfModels(self, modelNames: Dict[str, str], generatorIds):
    # Declare both models TF graphs and sessions
    tfGraphs = {
      FREQUENCY: tf.Graph(),
      COST: tf.Graph(),
    }
    self._tfSessions[FREQUENCY] = tf.compat.v1.Session(graph=tfGraphs[FREQUENCY])
    self._tfSessions[COST] = tf.compat.v1.Session(graph=tfGraphs[COST])

    tfSavers = {}
    # Recreate agents within each graph
    for modelType in MODEL_TYPES:
      self._allAgents[modelType]: List[Agent] = [Agent(id_, self._modelAdapters[modelType]) for id_ in generatorIds]
      tfSavers[modelType] = tf.compat.v1.train.Saver()

    # Restore TF sessions
    for modelType in MODEL_TYPES:
      modelPath = getPathForModel(modelNames[modelType])
      tfSavers[modelType].restore(self._tfSessions[modelType], modelPath)

  def testAgents(self, electricalSystemSpecs: ElectricalSystemSpecs, modelNameFreq: str, modelNameCost: str, rewardFnCost=costRewardFunction, stepsToTest: int = 500, frequencyWeight=0.7):
    self._actionWeights = {
      FREQUENCY: frequencyWeight,
      COST: 1 - frequencyWeight,
    }

    # Recreate simulation environment
    elecSystem = ElectricalSystemFactory.create(electricalSystemSpecs)

    self.loadTfModels(
      modelNames={
        FREQUENCY: modelNameFreq,
        COST: modelNameCost,
      },
      generatorIds=[generator.id_ for generator in electricalSystemSpecs.generators],
    )

    # Store initial state for all adapters
    for modelType in MODEL_TYPES:
      self._modelAdapters[modelType].storeInitialState(
        elecSystem=elecSystem,
        allAgents=self._allAgents[modelType],
      )

    for stepIdx in range(stepsToTest):
      # Get all agents' actions
      allActions = self.calculateAllActions(elecSystem)

      # Run all actions
      self.executeAllActions(allActions, elecSystem)

      # Calculate earned rewards per objective
      self.calculateRewards(elecSystem)

    # Close tf sessions after using them
    for tfSession in self._tfSessions.values():
      tfSession.close()

    return elecSystem, self._allRewards, self._allRewardComponents

  def calculateAllActions(self, elecSystem):
    allActions = {}
    for modelType in MODEL_TYPES:
      modelAgents = self._allAgents[modelType]
      tfSession = self._tfSessions[modelType]
      modelAdapter = self._modelAdapters[modelType]

      modelAdapter.storePreActionStateReward(elecSystem) # Store preaction state
      modelStates = modelAdapter.observeStates(elecSystem=elecSystem, allAgents=modelAgents) # Observe states
      modelActions = [agent.runActorAction(tfSession, modelStates.get(agent.getId())) for agent in modelAgents] # Calculate actions
      modelActions = [action[0, 0] for action in modelActions]
      allActions[modelType] = modelActions

    return allActions

  def executeAllActions(self, allActions, elecSystem):
    # Combine actions
    allActionsCombined = []
    for actionIdx in range(allActions[FREQUENCY]):
      totalAction = 0
      for modelType in MODEL_TYPES:
        action = allActions[modelType][actionIdx]
        weightedAction = self._actionWeights[modelType]*action
        totalAction += weightedAction
      allActionsCombined.append(totalAction)

    # actionsCombined = [frequencyWeight*actionFreq + costWeight*actionCost for (actionFreq, actionCost) in zip(allActions[FREQUENCY], allActions[COST])]

    # Execute agents'actions (i.e. update the generators' power output)
    agentIds = [agent.getId() for agent in self._allAgents[FREQUENCY]] # Both agent collections have the same ids...
    generatorUpdates = [NodePowerUpdate(
        id_=agentId,
        deltaPower=action
      ) for (agentId, action) in zip(agentIds, allActionsCombined)]
    elecSystem.updateGenerators(generatorUpdates)

  def calculateRewards(self, elecSystem):
    for modelType in MODEL_TYPES:
        earnedReward, rewardComponents = self._modelAdapters[modelType].calculateReward(elecSystem)
        self._allRewards[modelType].append(earnedReward)
        self._allRewardComponents[modelType].append(rewardComponents)
