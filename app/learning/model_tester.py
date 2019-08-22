from typing import List
import tensorflow as tf

from electricity import ElectricalSystemFactory
from dto import ElectricalSystemSpecs, NodePowerUpdate

from .learning_agent import Agent

class ModelTester():
  @staticmethod
  def testAgents(electricalSystemSpecs: ElectricalSystemSpecs, allAgents: List[Agent], stepsToTest: int = 500):

    elecSystem = ElectricalSystemFactory.create(electricalSystemSpecs)
    allRewards = []

    tfInit = tf.global_variables_initializer()

    # Main TF loop
    with tf.Session() as tfSession:
      tfSession.run(tfInit)

      # Test for 1000 steps
      for stepIdx in range(stepsToTest):
        # Get all agents' actions
        deltaFreqOriginal = elecSystem.getCurrentDeltaF()
        allActions = [agent.runActorAction(tfSession, deltaFreqOriginal) for agent in allAgents]
        allActions = [action[0, 0] for action in allActions]

        # Execute agents'actions (i.e. update the generators' power output)
        agentIds = [agent.getId() for agent in allAgents]
        generatorUpdates = [NodePowerUpdate(
            id_=agentId,
            deltaPower=action
          ) for (agentId, action) in zip(agentIds, allActions)]
        elecSystem.updateGenerators(generatorUpdates)

        deltaFreqNew = elecSystem.getCurrentDeltaF()
        earnedReward = 2**(10-abs(deltaFreqNew)) # TODO Calculate reward according to a given strategy
        allRewards.append(earnedReward)

    return elecSystem, allRewards
