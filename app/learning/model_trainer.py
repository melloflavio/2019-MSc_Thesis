# from pprint import pformat
import json
import tensorflow as tf
import numpy as np

from electricity import ElectricalSystem
from dto import NodePowerUpdate

from .learning_agent import Agent
from .learning_state import LearningState
from .learning_params import LearningParams
from .experience_buffer import ExperienceBuffer, LearningExperience

class ModelTrainer():
  @staticmethod
  def trainAgents(electricalSystem: ElectricalSystem):
    # Initialize Learning State
    LearningState().initData(
        allAgents=[Agent(_id) for _id in electricalSystem.getGeneratorIds()],
        xpBuffer=ExperienceBuffer()
    )

    # alias for quicker access
    _episode = LearningState().episode
    _model = LearningState().model
    _params = LearningParams()

    tfInit = tf.global_variables_initializer()

    # Main TF loop
    with tf.Session() as tfSession:
      tfSession.run(tfInit)

      # Run all learning episodes
      for episodeIdx in range(_params.numEpisodes):

        # Clear values regarding episodes
        ModelTrainer.resetEpisodeState()

        # Iterate over all the steps
        for stepIdx in range(_params.maxSteps):

          # Get all agents' actions
          currentDeltaF = electricalSystem.getCurrentDeltaF()
          actions = [agent.getActorAction(tfSession, currentDeltaF) for agent in _model.allAgents]

          # Execute agents'actions (i.e. update the generators' power output)
          agentIds = [agent.getId() for agent in _model.allAgents]
          generatorUpdates = [NodePowerUpdate(
              id_=agentId,
              deltaPower=action
            ) for (agentId, action) in zip(agentIds, actions)]
          electricalSystem.updateGenerators(generatorUpdates)

          newDeltaF = electricalSystem.getCurrentDeltaF()
          currentReward = 2**(10-abs(newDeltaF)) # TODO Calculate reward according to a given strategy
          _episode.cummReward += currentReward

          experience = LearningExperience(
              originalState     = currentDeltaF,
              destinationState  = newDeltaF,
              actions           = {agentId: action for (agentId, action) in zip(agentIds, actions)},
              reward            = currentReward,
          )
          _episode.experiences.append(experience)

          print(f'e{episodeIdx}s{stepIdx}: {json.dumps(experience._asdict(), indent=2)}')

  @staticmethod
  def resetEpisodeState():
    # Push current reward to reward list
    if (LearningState().episode.cummReward is not None):
      LearningState().model.cummRewardList.append(LearningState().episode.cummReward)

    # Clear episode values
    LearningState().episode.cummReward = 0
    LearningState().episode.episodeBuffer = []
