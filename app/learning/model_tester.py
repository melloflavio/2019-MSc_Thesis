from typing import List
import tensorflow as tf

from electricity import ElectricalSystemFactory
from dto import ElectricalSystemSpecs, NodePowerUpdate

from .learning_agent import Agent

class ModelTester():
  @staticmethod
  def testAgents(electricalSystemSpecs: ElectricalSystemSpecs, allAgents: List[Agent]):

    elecSystem = ElectricalSystemFactory.create(electricalSystemSpecs)

    tfInit = tf.global_variables_initializer()

    # Main TF loop
    with tf.Session() as tfSession:
      tfSession.run(tfInit)

      # Test for 1000 steps
      for stepIdx in range(1000):
        # Get all agents' actions
        currentDeltaF = elecSystem.getCurrentDeltaF()
        allActions = [agent.runActorAction(tfSession, currentDeltaF) for agent in allAgents]
        allActions = [action[0, 0] for action in allActions]

        # Execute agents'actions (i.e. update the generators' power output)
        agentIds = [agent.getId() for agent in allAgents]
        generatorUpdates = [NodePowerUpdate(
            id_=agentId,
            deltaPower=action
          ) for (agentId, action) in zip(agentIds, allActions)]
        elecSystem.updateGenerators(generatorUpdates)

      print("Finished testing")

    return elecSystem
