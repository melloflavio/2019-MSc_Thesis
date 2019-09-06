from typing import List
import tensorflow as tf

from electricity import ElectricalSystemFactory
from dto import ElectricalSystemSpecs, NodePowerUpdate
from models import getPathForModel

from .cost_reward import costRewardFunction
from .cost_learning_agent import CostAgent as Agent

class CostModelTester():
  @staticmethod
  def testAgents(electricalSystemSpecs: ElectricalSystemSpecs, modelName: str, rewardFn=costRewardFunction, stepsToTest: int = 500):
    # Clear existing graph
    tf.compat.v1.reset_default_graph()

    allRewards = [] # Used to plot reward history

    # Main TF loop
    with tf.compat.v1.Session() as tfSession:
      # Recreate the testing environment/TF variable placeholders
      elecSystem = ElectricalSystemFactory.create(electricalSystemSpecs)
      allAgents: List[Agent] = [Agent(generator.id_) for generator in electricalSystemSpecs.generators]
      _initTotalZ = sum(elecSystem.getGeneratorsOutputs().values())

      tfSaver = tf.compat.v1.train.Saver() # Saver obj used to restore session
      modelPath = getPathForModel(modelName)

      tfSaver.restore(tfSession, modelPath)

      # Test for 1000 steps
      for stepIdx in range(stepsToTest):
        # Get all agents' actions
        generatorsOutputsOrigin = elecSystem.getGeneratorsOutputs()

        allStatesOrigin = {actorId: {'genOutput': output, 'totalOutput':_initTotalZ} for actorId, output in generatorsOutputsOrigin.items()}
        allActions = [agent.runActorAction(tfSession, allStatesOrigin.get(agent.getId())) for agent in allAgents]
        allActions = [action[0, 0] for action in allActions]

        # Execute agents'actions (i.e. update the generators' power output)
        agentIds = [agent.getId() for agent in allAgents]
        generatorUpdates = [NodePowerUpdate(
            id_=agentId,
            deltaPower=action
          ) for (agentId, action) in zip(agentIds, allActions)]
        elecSystem.updateGenerators(generatorUpdates)

        generatorsOutputsDestination = elecSystem.getGeneratorsOutputs()
        totalOutputDestination = sum(generatorsOutputsDestination.values())

        totalCost = elecSystem.getTotalCost()

        costDifferential = elecSystem.getCostOptimalDiferential()
        earnedReward = rewardFn(totalOutputTarget=_initTotalZ,  totalOutputDestination=totalOutputDestination, totalCost=totalCost)
        allRewards.append(earnedReward)

    return elecSystem, allRewards
