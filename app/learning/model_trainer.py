# from pprint import pformat
import json
import tensorflow as tf
import numpy as np

from electricity import ElectricalSystemFactory
from dto import ElectricalSystemSpecs, NodePowerUpdate

from .learning_agent import Agent
from .learning_state import LearningState
from .learning_params import LearningParams
from .experience_buffer import ExperienceBuffer, LearningExperience
from .experience_buffer_dto import XpMiniBatch
from .critic_dto import CriticEstimateInput, CriticUpdateInput, CriticGradientInput
from .actor_dto import ActorUpdateInput

class ModelTrainer():
  @staticmethod
  def trainAgents(electricalSystemSpecs: ElectricalSystemSpecs):
    # Initialize Learning State
    LearningState().initData(
        allAgents=[Agent(generator.id_) for generator in electricalSystemSpecs.generators],
        xpBuffer=ExperienceBuffer(),
        epsilon=LearningParams().epsilon,
        electricalSystemSpecs=electricalSystemSpecs,
    )

    # alias for quicker access
    _episode = LearningState().episode
    _model = LearningState().model
    _params = LearningParams()

    tfInit = tf.global_variables_initializer()

    # Main TF loop
    with tf.Session() as tfSession:
      tfSession.run(tfInit)

      for episodeIdx in range(_params.numEpisodes):

        # Clear values regarding episodes
        ModelTrainer.resetEpisodeState()

        for stepIdx in range(_params.maxSteps):

          experience = ModelTrainer.executeStep(tfSession)
          print(f'e{episodeIdx}s{stepIdx}: {json.dumps(experience._asdict(), indent=2)}')

          # Update the model
          if (ModelTrainer.shouldUpdateModels(stepIdx)):
            ModelTrainer.runUpdateCycle(tfSession)

          # Update epsilom
          _model.epsilon = _model.epsilon*0.99999 if _model.epsilon < 0.5 else _model.epsilon*0.999999 #TODO isolate epsilon decay & parametrize

        # Store episodes' experiences if they are large enough (disconsider espisodes that ended prematurely)
        if len(_episode.experiences) >= 8:
            _model.xpBuffer.add(_episode.experiences)

    return _model.allAgents

  @staticmethod
  def executeStep(tfSession: tf.Session):
    _episode = LearningState().episode
    # Get all agents' actions
    originalDeltaF = _episode.electricalSystem.getCurrentDeltaF()
    allActions = ModelTrainer._01_calculateAllActorActions(tfSession, originalDeltaF)

    # Execute agents'actions (i.e. update the generators' power output)
    ModelTrainer._02_executeAllActorActions(allActions)

    # Calculate the earned reward
    newDeltaF = _episode.electricalSystem.getCurrentDeltaF()
    earnedReward = 2**(10-abs(newDeltaF)) # TODO Calculate reward according to a given strategy

    experience = ModelTrainer._03_storeEpisodeExperience(originalDeltaF, newDeltaF,allActions, earnedReward)
    return experience

  @staticmethod
  def shouldUpdateModels(stepIdx):
    numStoredEpisodes = LearningState().model.xpBuffer.numStoredEpisodes
    shouldUpdate = (
      stepIdx % 4 == 0  # Every N steps
      and numStoredEpisodes > 0 # Starting from the second episode (must have at least one full episode in xp buffer)
     )
    return shouldUpdate

  @staticmethod
  def resetEpisodeState():
    # Push current reward to reward list
    if (LearningState().episode.cummReward is not None):
      LearningState().model.cummRewardList.append(LearningState().episode.cummReward)

    # Clear episode values
    LearningState().episode.cummReward = 0
    LearningState().episode.episodeBuffer = []

    # Instantiate new slightly randomized electrical system
    specs = LearningState().model.electricalSystemSpecs
    LearningState().episode.electricalSystem = ElectricalSystemFactory.create(specs)

  @staticmethod
  def getEmptyLtsmState():
    """Generates an empty training state"""
    batchSize = LearningParams().batchSize
    lstmSize = LearningParams().nnShape.layer_00_ltsm
    emptyState = (np.zeros([batchSize, lstmSize]), ) * len(LearningState().model.allAgents)
    return emptyState

  @staticmethod
  def runUpdateCycle(tfSession: tf.Session):
    """Updates all agents' models using experiences previously stored in the experience buffer"""
    # Sample the experience batch (mini batch)
    xpBatch = ModelTrainer._04_sampleTrainingExperienceMiniBatch()

    # Get Target Actors' Actions
    allTargetActions = ModelTrainer._05_calculateTargetActionsForBatch(tfSession, xpBatch)

    # Get Target Critics' Q Estimations
    allCriticTargets = ModelTrainer._06_calculateTargetQvalsForBatch(tfSession, xpBatch, allTargetActions)

    # Update the critic networks with the new Q's
    ModelTrainer._07_updateCriticModelsForBatch(tfSession, xpBatch, allCriticTargets)

    # Calculate actions for all actors
    allNewActions = ModelTrainer._08_calculateActorActionsForBatch(tfSession, xpBatch)

    # Calculate the critic's gradients from the estimated actions
    allGradients = ModelTrainer._09_calculateCriticGradientsForBatch(tfSession, xpBatch, allNewActions)

    # Update the actor models with the gradients calculated by the critics
    ModelTrainer._10_updateActorModelsForBatch(tfSession, xpBatch, allGradients)

    # Update target actor and critic models for all agents
    ModelTrainer._11_updateTargetModels(tfSession)

  @staticmethod
  def _01_calculateAllActorActions(tfSession: tf.Session, currentDeltaF):
    _model = LearningState().model
    allActions = [agent.runActorAction(tfSession, currentDeltaF) for agent in _model.allAgents]
    allActions = [action[0, 0] + _model.epsilon * np.random.normal(0.0, 0.4) for action in allActions]
    return allActions

  @staticmethod
  def _02_executeAllActorActions(allActions):
    agentIds = [agent.getId() for agent in LearningState().model.allAgents]
    generatorUpdates = [NodePowerUpdate(
        id_=agentId,
        deltaPower=action
      ) for (agentId, action) in zip(agentIds, allActions)]
    LearningState().episode.electricalSystem.updateGenerators(generatorUpdates)

  @staticmethod
  def _03_storeEpisodeExperience(originalDeltaF, newDeltaF, allActions, earnedReward):
    agentIds = [agent.getId() for agent in LearningState().model.allAgents]
    LearningState().episode.cummReward += earnedReward
    experience = LearningExperience(
        originalState     = originalDeltaF,
        destinationState  = newDeltaF,
        actions           = {agentId: action for (agentId, action) in zip(agentIds, allActions)},
        reward            = earnedReward,
    )
    LearningState().episode.experiences.append(experience)
    return experience

  @staticmethod
  def _04_sampleTrainingExperienceMiniBatch() -> XpMiniBatch:
    batchSize = LearningParams().batchSize
    traceLength = LearningParams().traceLength
    xpBatch = LearningState().model.xpBuffer.getSample(batchSize, traceLength)

    return xpBatch

  @staticmethod
  def _05_calculateTargetActionsForBatch(tfSession: tf.Session, xpBatch: XpMiniBatch):
    allAgents = LearningState().model.allAgents
    allTargetActions = [
      agent.getActorTargetAction(
          tfSession=tfSession,
          state=xpBatch.destinationStates,
          ltsmState=ModelTrainer.getEmptyLtsmState(),
        )[0] # method returns tuple (action, nextState) here we only want the action
      for agent in allAgents]

    return allTargetActions

  @staticmethod
  def _06_calculateTargetQvalsForBatch(tfSession: tf.Session, xpBatch: XpMiniBatch, allTargetActions):
    allAgents = LearningState().model.allAgents
    batchSize = LearningParams().batchSize
    traceLength = LearningParams().traceLength
    gamma = LearningParams().gamma

    allCriticTargets = []
    for agentIdx, agent in enumerate(allAgents):
      targetAction = allTargetActions[agentIdx]
      otherTargetActionLists = [action for i, action in enumerate(allTargetActions) if i != agentIdx] # Get action elements other than this agent's
      otherTargetActions = [action for actionList in otherTargetActionLists for action in actionList] # Stack all other agents' actions in a single vector
      targetQ = agent.getTargetCriticEstimatedQ(
          tfSession=tfSession,
          criticIn=CriticEstimateInput(
              state=xpBatch.destinationStates,
              actionActor=targetAction,
              actionsOthers=otherTargetActions,
              ltsmInternalState=ModelTrainer.getEmptyLtsmState(),
              batchSize=batchSize,
              traceLength=traceLength,
          )
      )
      # Update Targets
      targetQ = xpBatch.rewards + gamma*targetQ

      allCriticTargets.append(targetQ)

    return allCriticTargets

  @staticmethod
  def _07_updateCriticModelsForBatch(tfSession: tf.Session, xpBatch: XpMiniBatch, allCriticTargets):
    allAgents = LearningState().model.allAgents
    batchSize = LearningParams().batchSize
    traceLength = LearningParams().traceLength

    for agentIdx, agent in enumerate(allAgents):
      agentActions = xpBatch.groupedActions.get(agent.getId())
      actionsOthers = {agentId:xpBatch.groupedActions[agentId] for agentId in xpBatch.groupedActions if agentId != agent.getId()} # Remove this agent from list
      actionsOthers = [action for actionList in actionsOthers.values() for action in actionList] # Stack all actions in a single array
      targetQs = allCriticTargets[agentIdx]
      agent.updateCritic(
          tfSession=tfSession,
          criticUpd=CriticUpdateInput(
              state=xpBatch.originalStates,
              actionActor=agentActions,
              actionsOthers=actionsOthers,
              targetQs=targetQs,
              ltsmInternalState=ModelTrainer.getEmptyLtsmState(),
              batchSize=batchSize,
              traceLength=traceLength,
        ),
      )

  @staticmethod
  def _08_calculateActorActionsForBatch(tfSession: tf.Session, xpBatch: XpMiniBatch):
    allAgents = LearningState().model.allAgents

    allNewActions = [agent.predictActorAction(
                tfSession=tfSession,
                currentDeltaF=xpBatch.originalStates,
                ltsmState=ModelTrainer.getEmptyLtsmState(),
              ) for agent in allAgents]

    return allNewActions

  @staticmethod
  def _09_calculateCriticGradientsForBatch(tfSession: tf.Session, xpBatch: XpMiniBatch, allNewActions):
    allAgents = LearningState().model.allAgents
    batchSize = LearningParams().batchSize
    traceLength = LearningParams().traceLength

    allGradients = []
    for agentIdx, agent in enumerate(allAgents):
      agentActions = allNewActions[agentIdx]
      otherActionLists = [action for i, action in enumerate(allNewActions) if i != agentIdx] # Get action elements other than this agent's
      actionsOthers = [action for actionList in otherActionLists for action in actionList] # Stack all other agents' actions in a single vector
      gradient = agent.calculateCriticGradients(
          tfSession=tfSession,
          inpt=CriticGradientInput(
              state=xpBatch.originalStates,
              actionActor=agentActions,
              actionsOthers=actionsOthers,
              ltsmInternalState=ModelTrainer.getEmptyLtsmState(),
              batchSize=batchSize,
              traceLength=traceLength,
          )
        )
      allGradients.append(gradient)
    return allGradients

  @staticmethod
  def _10_updateActorModelsForBatch(tfSession: tf.Session, xpBatch: XpMiniBatch, allGradients):
    allAgents = LearningState().model.allAgents
    batchSize = LearningParams().batchSize
    traceLength = LearningParams().traceLength

    for agentIdx, agent in enumerate(allAgents):
      agent.updateActor(
          tfSession=tfSession,
          inpt=ActorUpdateInput(
              state=xpBatch.originalStates,
              gradients=allGradients[agentIdx],
              ltsmInternalState=ModelTrainer.getEmptyLtsmState(),
              batchSize=batchSize,
              traceLength=traceLength,
            )
        )

  @staticmethod
  def _11_updateTargetModels(tfSession):
    allAgents = LearningState().model.allAgents

    for agent in allAgents:
      agent.updateTargetModels(tfSession)
