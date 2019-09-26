from typing import List
import tensorflow as tf

from electricity import ElectricalSystemFactory
from dto import ElectricalSystemSpecs, NodePowerUpdate
from models import getPathForModel

from .learning_agent import Agent
from .model_adapter import ModelAdapter

class ModelTester():
  def __init__(self, modelAdapter: ModelAdapter):
    self._modelAdapter: ModelAdapter = modelAdapter

  def testAgents(self, electricalSystemSpecs: ElectricalSystemSpecs, modelName: str, stepsToTest: int = 500):
    # Clear existing graph
    tf.compat.v1.reset_default_graph()

    # Recreate the testing environment/TF variable placeholders
    elecSystem = ElectricalSystemFactory.create(electricalSystemSpecs)
    allAgents: List[Agent] = [Agent(generator.id_, self._modelAdapter) for generator in electricalSystemSpecs.generators]
    self._modelAdapter.storeInitialState(
      elecSystem=elecSystem,
      allAgents=allAgents,
    )
    # _initTotalZ = sum(elecSystem.getGeneratorsOutputs().values())

    allRewards = [] # Used to plot reward history

    tfSaver = tf.compat.v1.train.Saver() # Saver obj used to restore session
    modelPath = getPathForModel(modelName)

    # Main TF loop
    with tf.compat.v1.Session() as tfSession:
      tfSaver.restore(tfSession, modelPath)

      # Test for 1000 steps
      for stepIdx in range(stepsToTest):
        # Get all agents' actions
        allStatesOrigin = self._modelAdapter.observeStates(elecSystem=elecSystem, allAgents=allAgents)

        allActions = [agent.runActorAction(tfSession, allStatesOrigin.get(agent.getId())) for agent in allAgents]
        allActions = [action[0, 0] for action in allActions]

        # Execute agents'actions (i.e. update the generators' power output)
        agentIds = [agent.getId() for agent in allAgents]
        generatorUpdates = [NodePowerUpdate(
            id_=agentId,
            deltaPower=action
          ) for (agentId, action) in zip(agentIds, allActions)]
        elecSystem.updateGenerators(generatorUpdates)

        # Calculate reward
        earnedReward, rewardComponents = self._modelAdapter.calculateReward(elecSystem)
        allRewards.append(earnedReward)

    return elecSystem, allRewards
