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

      # Run all learning episodes
      for episodeIdx in range(_params.numEpisodes):

        # Clear values regarding episodes
        ModelTrainer.resetEpisodeState()

        # Iterate over all the steps
        for stepIdx in range(_params.maxSteps):

          # Get all agents' actions
          currentDeltaF = _episode.electricalSystem.getCurrentDeltaF()
          allActions = [agent.runActorAction(tfSession, currentDeltaF) for agent in _model.allAgents]
          allActions = [action[0, 0] + _model.epsilon * np.random.normal(0.0, 0.4) for action in allActions]

          # Execute agents'actions (i.e. update the generators' power output)
          agentIds = [agent.getId() for agent in _model.allAgents]
          generatorUpdates = [NodePowerUpdate(
              id_=agentId,
              deltaPower=action
            ) for (agentId, action) in zip(agentIds, allActions)]
          _episode.electricalSystem.updateGenerators(generatorUpdates)

          newDeltaF = _episode.electricalSystem.getCurrentDeltaF()
          currentReward = 2**(10-abs(newDeltaF)) # TODO Calculate reward according to a given strategy
          _episode.cummReward += currentReward

          experience = LearningExperience(
              originalState     = currentDeltaF,
              destinationState  = newDeltaF,
              actions           = {agentId: action for (agentId, action) in zip(agentIds, allActions)},
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
              rewards) = _model.xpBuffer.getSample(_params.batchSize, _params.traceLength)

            # Get Target Actors' Actions
            allTargetActions = [agent.getActorTargetAction(
                tfSession=tfSession,
                state=destinationStates,
                ltsmState=ModelTrainer.getEmptyLtsmState(),
            )[0] # method returns tuple (action, nextState) here we only want the action
            for agent in _model.allAgents]

            # Get Target Critics' Q Estimations
            allCriticTargets = []
            for agentIdx, agent in enumerate(_model.allAgents):
              targetAction = allTargetActions[agentIdx]
              otherTargetActionLists = [action for i, action in enumerate(allTargetActions) if i != agentIdx] # Get action elements other than this agent's
              otherTargetActions = [action for actionList in otherTargetActionLists for action in actionList] # Stack all other agents' actions in a single vector
              targetQ = agent.getTargetCriticEstimatedQ(
                  tfSession=tfSession,
                  criticIn=CriticEstimateInput(
                      state=destinationStates,
                      actionActor=targetAction,
                      actionsOthers=otherTargetActions,
                      ltsmInternalState=ModelTrainer.getEmptyLtsmState(),
                      batchSize=_params.batchSize,
                      traceLength=_params.traceLength,
                  )
              )
              # Update Targets
              targetQ = rewards + _params.gamma*targetQ

              allCriticTargets.append(targetQ)

            # Update the critic networks with the new Q's
            for agentIdx, agent in enumerate(_model.allAgents):
              agentActions = groupedActions.get(agent.getId())
              actionsOthers = {agentId:groupedActions[agentId] for agentId in groupedActions if agentId != agent.getId()} # Remove this agent from list
              actionsOthers = [action for actionList in actionsOthers.values() for action in actionList] # Stack all actions in a single array
              targetQs = allCriticTargets[agentIdx]
              agent.updateCritic(
                  tfSession=tfSession,
                  criticUpd=CriticUpdateInput(
                      state=originalStates,
                      actionActor=agentActions,
                      actionsOthers=actionsOthers,
                      targetQs=targetQs,
                      ltsmInternalState=ModelTrainer.getEmptyLtsmState(),
                      batchSize=_params.batchSize,
                      traceLength=_params.traceLength,
                ),
              )

            allNewActions = [agent.predictActorAction(
                tfSession=tfSession,
                currentDeltaF=originalStates,
                ltsmState=ModelTrainer.getEmptyLtsmState(),
              ) for agent in _model.allAgents]

            allGradients = []
            for agentIdx, agent in enumerate(_model.allAgents):
              agentActions = allNewActions[agentIdx]
              otherActionLists = [action for i, action in enumerate(allNewActions) if i != agentIdx] # Get action elements other than this agent's
              actionsOthers = [action for actionList in otherActionLists for action in actionList] # Stack all other agents' actions in a single vector
              gradient = agent.calculateCriticGradients(
                  tfSession=tfSession,
                  inpt=CriticGradientInput(
                      state=originalStates,
                      actionActor=agentActions,
                      actionsOthers=actionsOthers,
                      ltsmInternalState=ModelTrainer.getEmptyLtsmState(),
                      batchSize=_params.batchSize,
                      traceLength=_params.traceLength,
                  )
                )
              allGradients.append(gradient)

            for agentIdx, agent in enumerate(_model.allAgents):
              agent.updateActor(
                  tfSession=tfSession,
                  inpt=ActorUpdateInput(
                      state=originalStates,
                      gradients=allGradients[agentIdx],
                      ltsmInternalState=ModelTrainer.getEmptyLtsmState(),
                      batchSize=_params.batchSize,
                      traceLength=_params.traceLength,
                    )
                )

            for agent in _model.allAgents:
              agent.updateTargetModels(tfSession)


          _model.epsilon = _model.epsilon*0.99999 if _model.epsilon < 0.5 else _model.epsilon*0.999999 #TODO isolate epsilon decay & parametrize


        if len(_episode.experiences) >= 8:
            _model.xpBuffer.add(_episode.experiences)

    agents = {agent.getId(): agent for agent in _model.allAgents}
    return agents

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
