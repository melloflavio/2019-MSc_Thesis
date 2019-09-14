import json
import tensorflow as tf
import numpy as np

from electricity import ElectricalSystemFactory
from dto import ElectricalSystemSpecs, NodePowerUpdate
from models import getPathForModel, getPathForParams

from .learning_agent import Agent
from .reward import rewardFunction
from ..learning_state import LearningState
from ..learning_params import LearningParams
from ..experience_buffer import ExperienceBuffer, LearningExperience
from ..experience_buffer_dto import XpMiniBatch
from ..critic_dto import CriticEstimateInput, CriticUpdateInput, CriticGradientInput
from ..actor_dto import ActorUpdateInput
from ..epsilon import Epsilon

class ModelTrainer():
  _rewardFn = None
  def __init__(self, rewardFn=rewardFunction):
    self._rewardFn = rewardFn

  def trainAgents(self):
    # Clears existing TF graph
    tf.compat.v1.reset_default_graph()

    # Initialize Learning State
    LearningState().initData(
        allAgents=[Agent(generator.id_) for generator in LearningParams().electricalSystemSpecs.generators],
        xpBuffer=ExperienceBuffer(),
        epsilon=Epsilon(
            specs=LearningParams().epsilonSpecs,
            numEpisodes=LearningParams().numEpisodes,
            stepsPerEpisode=LearningParams().maxSteps,
        ),
    )

    # alias for quicker access
    _episode = LearningState().episode
    _model = LearningState().model
    _params = LearningParams()

    tfInit = tf.compat.v1.global_variables_initializer()

    # Main TF loop
    with tf.compat.v1.Session() as tfSession:
      tfSession.run(tfInit)

      print(f'Training model: {LearningParams().modelName} - ', end='')
      for episodeIdx in range(_params.numEpisodes):

        # Print progress every 5%
        if (episodeIdx%(_params.numEpisodes*5/100) == 0):
          progressPrcent = round((episodeIdx/_params.numEpisodes)*100)
          print(f'{progressPrcent}% ', end='')

        # Clear values regarding episodes
        self.resetEpisodeState(episodeIdx)

        for stepIdx in range(_params.maxSteps):

          experience = self.executeStep(tfSession)
          # print(f'e{episodeIdx}s{stepIdx}: {json.dumps(experience._asdict(), indent=2)}')

          # Update the model
          if (self.shouldUpdateModels(stepIdx)):
            self.runUpdateCycle(tfSession)

          # Update epsilom
          _model.epsilon.decay()

          # End episode prematurely if things diverge too much
          if (self._shouldStopEarly(_episode.electricalSystem)):
            break

        # Store episodes' experiences if they are large enough to have at least a single complete trace
        if len(_episode.experiences) >= LearningParams().traceLength:
            _model.xpBuffer.add(_episode.experiences)

      # Save complete model in form of tensorflow session
      print('100%')
      self.saveModels(tfSession)
    return _model.allAgents

  def executeStep(self, tfSession: tf.compat.v1.Session):
    _episode = LearningState().episode
    # Get all agents' actions
    deltaFreqOriginal = _episode.electricalSystem.getCurrentDeltaF()
    generatorsOutputsOrigin = _episode.electricalSystem.getGeneratorsOutputs()
    totalOutputOrigin = sum(generatorsOutputsOrigin.values())
    allStatesOrigin = {actorId: {'genOutput': output, 'totalOutput':totalOutputOrigin, 'deltaFreq':deltaFreqOriginal} for actorId, output in generatorsOutputsOrigin.items()}
    allActions = self._01_calculateAllActorActions(tfSession, allStatesOrigin)

    # Execute agents'actions (i.e. update the generators' power output)
    self._02_executeAllActorActions(allActions)

    # Calculate the earned reward
    deltaFreqDestination = _episode.electricalSystem.getCurrentDeltaF()
    generatorsOutputsDestination = _episode.electricalSystem.getGeneratorsOutputs()
    totalOutputDestination = sum(generatorsOutputsDestination.values())
    allStatesDestination = {actorId: {'genOutput': output, 'totalOutput':totalOutputDestination, 'deltaFreq':deltaFreqDestination} for actorId, output in generatorsOutputsDestination.items()}

    totalCost = _episode.electricalSystem.getTotalCost()
    earnedReward, rewardComponents = self._rewardFn(deltaFreq=deltaFreqDestination, totalCost=totalCost)

    experience = self._03_storeEpisodeExperience(allStatesOrigin, allStatesDestination, allActions, earnedReward, rewardComponents)
    return experience

  def shouldUpdateModels(self, stepIdx):
    numStoredEpisodes = LearningState().model.xpBuffer.numStoredEpisodes
    shouldUpdate = (
      stepIdx % LearningParams().batchSize == 0  # Every N steps (same as batch size)
      and numStoredEpisodes > 0 # Starting from the second episode (must have at least one full episode in xp buffer)
     )
    return shouldUpdate

  def resetEpisodeState(self, episodeIdx: int):
    # Push current reward to reward list
    if (LearningState().episode.cummReward is not None):
      LearningState().model.cummRewardList.append(LearningState().episode.cummReward)

    # Store a snapshop of the rewards every 10%
    episodeRewardDetails = LearningState().episode.allRewards
    if (episodeIdx % (LearningParams().numEpisodes/10) == 0 and episodeRewardDetails):
      LearningState().model.allRewards.append(episodeRewardDetails)


    # Clear episode values
    LearningState().episode.cummReward = 0
    LearningState().episode.experiences = []
    LearningState().episode.allRewards=[]

    # Instantiate new slightly randomized electrical system
    specs = LearningParams().electricalSystemSpecs
    LearningState().episode.electricalSystem = ElectricalSystemFactory.create(specs)

    # Refresh LTSM states for all actors
    for agent in LearningState().model.allAgents:
      agent.resetLtsmState()

    # self._initTotalZ = sum(LearningState().episode.electricalSystem.getGeneratorsOutputs().values())

  def getEmptyLtsmState(self):
    """Generates an empty training state"""
    batchSize = LearningParams().batchSize
    lstmSize = LearningParams().nnShape.layer_00_ltsm
    emptyState = (np.zeros([batchSize, lstmSize]), ) * len(LearningState().model.allAgents)
    return emptyState

  def runUpdateCycle(self, tfSession: tf.compat.v1.Session):
    """Updates all agents' models using experiences previously stored in the experience buffer"""
    # Sample the experience batch (mini batch)
    xpBatch = self._04_sampleTrainingExperienceMiniBatch()

    # Get Target Actors' Actions
    allTargetActions = self._05_calculateTargetActionsForBatch(tfSession, xpBatch)

    # Get Target Critics' Q Estimations
    allCriticTargets = self._06_calculateTargetQvalsForBatch(tfSession, xpBatch, allTargetActions)

    # Update the critic networks with the new Q's
    self._07_updateCriticModelsForBatch(tfSession, xpBatch, allCriticTargets)

    # Calculate actions for all actors
    allNewActions = self._08_calculateActorActionsForBatch(tfSession, xpBatch)

    # Calculate the critic's gradients from the estimated actions
    allGradients = self._09_calculateCriticGradientsForBatch(tfSession, xpBatch, allNewActions)

    # Update the actor models with the gradients calculated by the critics
    self._10_updateActorModelsForBatch(tfSession, xpBatch, allGradients)

    # Update target actor and critic models for all agents
    self._11_updateTargetModels(tfSession)

  def saveModels(self, tfSession: tf.compat.v1.Session):
    modelName = LearningParams().modelName
    #Save TF Models
    modelPath = getPathForModel(modelName)
    saver = tf.compat.v1.train.Saver()
    savedPath = saver.save(tfSession, modelPath)
    print(f'Model saved in path: {savedPath}')

    # Dump LearningParams to a json file as means of keeping history of the params used for this model's training
    paramsPath = getPathForParams(modelName)
    paramsJsonObj = json.loads(LearningParams().to_json())
    paramsJsonStr = json.dumps(paramsJsonObj, indent=4, sort_keys=True)
    with open(paramsPath, 'w') as outFile:
      outFile.write(paramsJsonStr)


  def _01_calculateAllActorActions(self, tfSession: tf.compat.v1.Session, allStates):
    _model = LearningState().model
    allActions = [agent.runActorAction(tfSession, allStates.get(agent.getId())) for agent in _model.allAgents]
    allActions = [action[0, 0] + _model.epsilon.value * np.random.normal(0.0, 0.4) for action in allActions]
    return allActions

  def _02_executeAllActorActions(self, allActions):
    agentIds = [agent.getId() for agent in LearningState().model.allAgents]
    generatorUpdates = [NodePowerUpdate(
        id_=agentId,
        deltaPower=action
      ) for (agentId, action) in zip(agentIds, allActions)]
    LearningState().episode.electricalSystem.updateGenerators(generatorUpdates)

  def _03_storeEpisodeExperience(self, allStatesOrigin, allStatesDestination, allActions, earnedReward, rewardComponents):
    agentIds = [agent.getId() for agent in LearningState().model.allAgents]
    LearningState().episode.cummReward += earnedReward
    LearningState().episode.allRewards.append(rewardComponents)
    experience = LearningExperience(
        originalState     = allStatesOrigin,
        destinationState  = allStatesDestination,
        actions           = {agentId: action for (agentId, action) in zip(agentIds, allActions)},
        reward            = earnedReward,
    )
    LearningState().episode.experiences.append(experience)
    return experience

  def _04_sampleTrainingExperienceMiniBatch(self) -> XpMiniBatch:
    batchSize = LearningParams().batchSize
    traceLength = LearningParams().traceLength
    xpBatch = LearningState().model.xpBuffer.getSample(batchSize, traceLength)

    return xpBatch

  def _05_calculateTargetActionsForBatch(self, tfSession: tf.compat.v1.Session, xpBatch: XpMiniBatch):
    allAgents = LearningState().model.allAgents
    allTargetActions = [
      agent.peekActorTargetAction(
          tfSession=tfSession,
          state=xpBatch.destinationStates.get(agent.getId()),
          ltsmState=self.getEmptyLtsmState(),
        )
      for agent in allAgents]

    return allTargetActions

  def _06_calculateTargetQvalsForBatch(self, tfSession: tf.compat.v1.Session, xpBatch: XpMiniBatch, allTargetActions):
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
              state=xpBatch.destinationStates.get(agent.getId()),
              actionActor=targetAction,
              actionsOthers=otherTargetActions,
              ltsmInternalState=self.getEmptyLtsmState(),
              batchSize=batchSize,
              traceLength=traceLength,
          )
      )
      # Update Targets
      targetQ = xpBatch.rewards + gamma*targetQ

      allCriticTargets.append(targetQ)

    return allCriticTargets

  def _07_updateCriticModelsForBatch(self, tfSession: tf.compat.v1.Session, xpBatch: XpMiniBatch, allCriticTargets):
    allAgents = LearningState().model.allAgents
    batchSize = LearningParams().batchSize
    traceLength = LearningParams().traceLength

    for agentIdx, agent in enumerate(allAgents):
      agentActions = xpBatch.allActions.get(agent.getId())
      actionsOthers = {agentId:xpBatch.allActions[agentId] for agentId in xpBatch.allActions.keys() if agentId != agent.getId()} # Remove this agent from list
      actionsOthers = [action for actionList in actionsOthers.values() for action in actionList] # Stack all actions in a single array
      targetQs = allCriticTargets[agentIdx]
      agent.updateCritic(
          tfSession=tfSession,
          criticUpd=CriticUpdateInput(
              state=xpBatch.originalStates.get(agent.getId()),
              actionActor=agentActions,
              actionsOthers=actionsOthers,
              targetQs=targetQs,
              ltsmInternalState=self.getEmptyLtsmState(),
              batchSize=batchSize,
              traceLength=traceLength,
        ),
      )

  def _08_calculateActorActionsForBatch(self, tfSession: tf.compat.v1.Session, xpBatch: XpMiniBatch):
    allAgents = LearningState().model.allAgents

    allNewActions = [agent.peekActorAction(
                tfSession=tfSession,
                currentDeltaF=xpBatch.originalStates.get(agent.getId()),
                ltsmState=self.getEmptyLtsmState(),
              ) for agent in allAgents]

    return allNewActions

  def _09_calculateCriticGradientsForBatch(self, tfSession: tf.compat.v1.Session, xpBatch: XpMiniBatch, allNewActions):
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
              state=xpBatch.originalStates.get(agent.getId()),
              actionActor=agentActions,
              actionsOthers=actionsOthers,
              ltsmInternalState=self.getEmptyLtsmState(),
              batchSize=batchSize,
              traceLength=traceLength,
          )
      )
      allGradients.append(gradient)
    return allGradients

  def _10_updateActorModelsForBatch(self, tfSession: tf.compat.v1.Session, xpBatch: XpMiniBatch, allGradients):
    allAgents = LearningState().model.allAgents
    batchSize = LearningParams().batchSize
    traceLength = LearningParams().traceLength

    for agentIdx, agent in enumerate(allAgents):
      agent.updateActor(
          tfSession=tfSession,
          inpt=ActorUpdateInput(
              state=xpBatch.originalStates.get(agent.getId()),
              gradients=allGradients[agentIdx],
              ltsmInternalState=self.getEmptyLtsmState(),
              batchSize=batchSize,
              traceLength=traceLength,
          )
      )

  def _11_updateTargetModels(self, tfSession):
    allAgents = LearningState().model.allAgents

    for agent in allAgents:
      agent.updateTargetModels(tfSession)

### BEGIN ABSTRACT METHODS
  def _shouldStopEarly(self, elecSystem):
    costDifferential = elecSystem.getCostOptimalDiferential()
    shouldStop = abs(costDifferential) > 50
    return shouldStop
