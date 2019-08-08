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

          # Update the model
          if (
            stepIdx % 4 == 0  # Every N steps
            and _model.xpBuffer.numStoredEpisodes > 0 # Starting from the second episode (must have at least one full episode in xp buffer)
          ):

            # Sample the experience batch (mini batch)
            ( originalStates,
              destinationStates,
              groupedActions,
              rewards) = _model.xpBuffer.getSample(_params.batchSize, _params.traceSize)

            # Reset the recurrent layer's hidden state and get states
            # stateTrain = (np.zeros([_params.batchSize, _params.nnShape.layer_00_ltsm]), ) * len(_model.allAgents)
            targetActions = [agent.getActorTargetAction(tfSession, destinationStates) for agent in _model.allAgents]
          # def getActorTargetAction(self, tfSession: tf.Session, deltaF):

        if len(_episode.experiences) >= 8:
            _model.xpBuffer.add(_episode.experiences)

  @staticmethod
  def resetEpisodeState():
    # Push current reward to reward list
    if (LearningState().episode.cummReward is not None):
      LearningState().model.cummRewardList.append(LearningState().episode.cummReward)

    # Clear episode values
    LearningState().episode.cummReward = 0
    LearningState().episode.episodeBuffer = []
