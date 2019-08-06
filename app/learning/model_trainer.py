import tensorflow as tf

from electricity import ElectricalSystem

from .learning_agent import Agent
from .learning_state import LearningState
from .learning_params import LearningParams
from .experience_buffer import ExperienceBuffer

class ModelTrainer():
  @staticmethod
  def trainAgents(electricalSystem: ElectricalSystem):
    # Initialize Agents according to generators present in electrical system
    LearningState().model.agents = [Agent(_id) for _id in electricalSystem.getGeneratorIds()]
    allAgents = LearningState().model.agents # alias for quicker access

    # Initialize xp buffer
    LearningState().model.xpBuffer = ExperienceBuffer()
    LearningState().model.cummRewardList = []

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
          currentDeltaF = electricalSystem.getCurrentDeltaF()

          actions = [agent.getActorAction(tfSession, currentDeltaF) for agent in allAgents]
          agentIds = [agent.getId() for agent in allAgents]




  @staticmethod
  def resetEpisodeState():
    # Push current reward to reward list
    if (LearningState().episode.cummReward is not None):
      LearningState().model.cummRewardList.append(LearningState().episode.cummReward)

    # Clear episode values
    LearningState().episode.cummReward = 0
    LearningState().episode.episodeBuffer = []
