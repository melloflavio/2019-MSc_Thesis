import tensorflow as tf
import numpy as np

from .actor import Actor
from .critic import Critic
from .learning_params import LearningParams
from .learning_state import LearningState
from .actor_dto import ActionInput, ActorUpdateInput
from .critic_dto import CriticEstimateInput, CriticUpdateInput, CriticGradientInput

class Agent():
  """Entity representing a single agent in the scenario to be learned. Contains the actors & critics associated with learning"""

  def __init__(self, _id, modelAdapter):
    self._id = _id # Unique identifier of the agent
    self.actor       = modelAdapter.Actor(scope=f'{modelAdapter.SCOPE_PREFIX}_{_id}_actor')
    self.actorTarget = modelAdapter.Actor(scope=f'{modelAdapter.SCOPE_PREFIX}_{_id}_actor_target')
    self.critic       = modelAdapter.Critic(scope=f'{modelAdapter.SCOPE_PREFIX}_{_id}_critic')
    self.criticTarget = modelAdapter.Critic(scope=f'{modelAdapter.SCOPE_PREFIX}_{_id}_critic_target')

    # Create Op Holders for target networks
    self.actorTarget.createOpHolder(self.actor.networkParams, LearningParams().tau)
    self.criticTarget.createOpHolder(self.critic.networkParams, LearningParams().tau)

    # Init execution and training LTSM states
    self._ltsmStateExecution = None
    self._ltsmStateTraining = None

    self.resetAllLtsmStates()

  def resetAllLtsmStates(self):
    # Execution is done one step at a time
    executionBatchSize = 1
    self._ltsmStateExecution = self.getEmptyLtsmState(executionBatchSize)

    # Training is done in batches, always with an empty ltsm state
    traningBatchSize = LearningParams().batchSize
    self._ltsmStateTraining = self.getEmptyLtsmState(traningBatchSize)

  def getEmptyLtsmState(self, batchSize):
    """Generates an empty LTSM state"""
    lstmSize = LearningParams().nnShape.layer_00_ltsm
    emptyState = (np.zeros([batchSize, lstmSize]), np.zeros([batchSize, lstmSize]))
    return emptyState

  def getId(self):
    return self._id

  def runActorAction(self, tfSession: tf.compat.v1.Session, currentDeltaF):
    (action, nextState) = self.actor.getAction(
        tfSession=tfSession,
        actionIn=ActionInput(
            actorInput=[[currentDeltaF]],
            ltsmInternalState=self._ltsmStateExecution,
            batchSize=1,
            traceLength=1,
        )
    )

    self._ltsmStateExecution = nextState

    return action

  def peekActorAction(self, tfSession: tf.compat.v1.Session, currentDeltaF):
    action = self.actor.getActionOnly(
        tfSession=tfSession,
        actionIn=ActionInput(
            actorInput=currentDeltaF,
            ltsmInternalState=self._ltsmStateTraining,
            batchSize=LearningParams().batchSize,
            traceLength=LearningParams().traceLength,
        )
    )

    return action

  def peekActorTargetAction(self, tfSession: tf.compat.v1.Session, state):
    action = self.actorTarget.getActionOnly(
        tfSession=tfSession,
        actionIn=ActionInput(
            actorInput=state,
            ltsmInternalState=self._ltsmStateTraining,
            batchSize=LearningParams().batchSize,
            traceLength=LearningParams().traceLength,
        )
    )

    return action

  def getTargetCriticEstimatedQ(self, tfSession: tf.compat.v1.Session, criticIn: CriticEstimateInput):
    estimatedQ = self.criticTarget.getEstimatedQ(
        tfSession=tfSession,
        criticIn=criticIn,
        ltsmState=self._ltsmStateTraining,
    )

    return estimatedQ

  def updateCritic(self, tfSession: tf.compat.v1.Session, criticUpd: CriticUpdateInput):
    self.critic.updateModel(
        tfSession=tfSession,
        criticUpd=criticUpd,
        ltsmState=self._ltsmStateTraining,
    )

  def calculateCriticGradients(self, tfSession: tf.compat.v1.Session, inpt: CriticGradientInput):
    gradients = self.critic.calculateGradients(
        tfSession=tfSession,
        inpt=inpt,
        ltsmState=self._ltsmStateTraining,
    )
    return gradients

  def updateActor(self, tfSession: tf.compat.v1.Session, inpt: ActorUpdateInput):
    self.actor.updateModel(
      tfSession=tfSession,
      inpt=inpt,
      ltsmState=self._ltsmStateTraining
    )

  def updateTargetModels(self, tfSession: tf.compat.v1.Session):
    self.actorTarget.updateNetParams(tfSession)
    self.criticTarget.updateNetParams(tfSession)
