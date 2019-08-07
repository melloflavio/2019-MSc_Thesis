from pprint import pprint
import tensorflow as tf

from electricity import ElectricalSystem
from dto import NodePowerUpdate

from .learning_agent import Agent
from .learning_state import LearningState
from .learning_params import LearningParams
from .experience_buffer import ExperienceBuffer, LearningExperience

class ModelTrainer():
  @staticmethod
  def trainAgents(electricalSystem: ElectricalSystem):
    _episode = LearningState().episode
    _model = LearningState().model # alias for quicker access

    # Initialize Agents according to generators present in electrical system
    _model.allAgents = [Agent(_id) for _id in electricalSystem.getGeneratorIds()]


    # Initialize xp buffer
    _model.xpBuffer = ExperienceBuffer()
    _model.cummRewardList = []

    tfInit = tf.global_variables_initializer()

    # Main TF loop
    with tf.Session() as tfSession:
      tfSession.run(tfInit)

      # Run all learning episodes
      for episodeIdx in range(LearningParams().numEpisodes):

        # Clear values regarding episodes
        ModelTrainer.resetEpisodeState()

        # Iterate over all the steps
        for stepIdx in range(LearningParams().maxSteps):

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
          currentReward = 1 # TODO Calculate reward according to a given strategy
          _episode.cummReward += currentReward

          experience = LearningExperience(
              originalState     = currentDeltaF,
              destinationState  = newDeltaF,
              actions           = {agentId: action for (agentId, action) in zip(agentIds, actions)},
              reward            = currentReward,
          )
          _episode.experiences.append(experience)

          pprint(dict(experience._asdict()))
          # experience = np.array([current_f,new_f,a_1,a_2,r])
          # episodeBuffer.append(experience)
          # print("Delta f: ",round(current_f,2)," A1: ",a_1," A2: ",a_2, " Reward: ",r)




  @staticmethod
  def resetEpisodeState():
    # Push current reward to reward list
    if (LearningState().episode.cummReward is not None):
      LearningState().model.cummRewardList.append(LearningState().episode.cummReward)

    # Clear episode values
    LearningState().episode.cummReward = 0
    LearningState().episode.episodeBuffer = []
