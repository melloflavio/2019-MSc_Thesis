from typing import List
import tensorflow as tf

from electricity import ElectricalSystemFactory
from dto import ElectricalSystemSpecs, NodePowerUpdate
from models import getPathForModel

from .cost_learning_agent import CostAgent as Agent

class CostModelTester():
  @staticmethod
  def testAgents(electricalSystemSpecs: ElectricalSystemSpecs, modelName: str, stepsToTest: int = 500):
    # Clear existing graph
    tf.compat.v1.reset_default_graph()

    # Recreate the testing environment/TF variable placeholders
    elecSystem = ElectricalSystemFactory.create(electricalSystemSpecs)
    allAgents: List[Agent] = [Agent(generator.id_) for generator in electricalSystemSpecs.generators]

    allRewards = [] # Used to plot reward history

    tfSaver = tf.compat.v1.train.Saver() # Saver obj used to restore session
    modelPath = getPathForModel(modelName)

    # Main TF loop
    with tf.compat.v1.Session() as tfSession:
      tfSaver.restore(tfSession, modelPath)

      # Test for 1000 steps
      for stepIdx in range(stepsToTest):
        # Get all agents' actions
        # deltaFreqOriginal = elecSystem.getCurrentDeltaF()
        generatorsOutputsOrigin = elecSystem.getGeneratorsOutputs()
        totalOutputOrigin = sum(generatorsOutputsOrigin.values())
        allStatesOrigin = {actorId: {'genOutput': output, 'totalOutput':totalOutputOrigin} for actorId, output in generatorsOutputsOrigin.items()}
        allActions = [agent.runActorAction(tfSession, allStatesOrigin.get(agent.getId())) for agent in allAgents]
        allActions = [action[0, 0] for action in allActions]

        # Execute agents'actions (i.e. update the generators' power output)
        agentIds = [agent.getId() for agent in allAgents]
        generatorUpdates = [NodePowerUpdate(
            id_=agentId,
            deltaPower=action
          ) for (agentId, action) in zip(agentIds, allActions)]
        elecSystem.updateGenerators(generatorUpdates)

        # deltaFreqNew = elecSystem.getCurrentDeltaF()
        costDifferential = elecSystem.getCostOptimalDiferential()
        earnedReward = 2**(10-abs(costDifferential)) # TODO Calculate reward according to a given strategy
        allRewards.append(earnedReward)

    return elecSystem, allRewards
