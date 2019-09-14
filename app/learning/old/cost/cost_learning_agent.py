import tensorflow as tf
import numpy as np

from .cost_actor import CostActorMaddpg as Actor
from .cost_critic import CostCriticMaddpg as Critic
from ..learning_params import LearningParams
from ..learning_state import LearningState
from ..actor_dto import ActionInput, ActorUpdateInput
from ..critic_dto import CriticEstimateInput, CriticUpdateInput, CriticGradientInput

_scopeSuffix='cost'

class CostAgent():
  """Entity representing a single agent in the scenario to be learned. Contains the actors & critics associated with learning"""

  def __init__(self, _id):
    self._id = _id # Unique identifier of the agent
    self.actor       = Actor(scope=f'{_scopeSuffix}_{_id}_actor')
    self.actorTarget = Actor(scope=f'{_scopeSuffix}_{_id}_actor_target')
    self.critic       = Critic(scope=f'{_scopeSuffix}_{_id}_critic')
    self.criticTarget = Critic(scope=f'{_scopeSuffix}_{_id}_critic_target')

    # Create Op Holders for target networks
    self.actorTarget.createOpHolder(self.actor.networkParams, LearningParams().tau)
    self.criticTarget.createOpHolder(self.critic.networkParams, LearningParams().tau)

    self.ltsmState = None
    self.resetLtsmState()

  def resetLtsmState(self):
    # Initial empty input state (must have size of LTSM)
    ltsmSize = LearningParams().nnShape.layer_00_ltsm
    ltsmState = (np.zeros([1, ltsmSize]), np.zeros([1, ltsmSize]))
    self.ltsmState = ltsmState

  def getId(self):
    return self._id

  def runActorAction(self, tfSession: tf.compat.v1.Session, currentDeltaF):
    (action, nextState) = self.actor.getAction(
        tfSession=tfSession,
        actionIn=ActionInput(
            actorInput=[[currentDeltaF]],
            ltsmInternalState=self.ltsmState,
            batchSize=1,
            traceLength=1,
        )
    )

    self.ltsmState = nextState

    return action

  def peekActorAction(self, tfSession: tf.compat.v1.Session, currentDeltaF, ltsmState):
    action = self.actor.getActionOnly(
        tfSession=tfSession,
        actionIn=ActionInput(
            actorInput=currentDeltaF,
            ltsmInternalState=ltsmState,
            batchSize=LearningParams().batchSize,
            traceLength=LearningParams().traceLength,
        )
    )

    return action

  def peekActorTargetAction(self, tfSession: tf.compat.v1.Session, state, ltsmState):
    action = self.actorTarget.getActionOnly(
        tfSession=tfSession,
        actionIn=ActionInput(
            actorInput=state,
            ltsmInternalState=ltsmState,
            batchSize=LearningParams().batchSize,
            traceLength=LearningParams().traceLength,
        )
    )

    return action

  def getTargetCriticEstimatedQ(self, tfSession: tf.compat.v1.Session, criticIn: CriticEstimateInput):
    estimatedQ = self.criticTarget.getEstimatedQ(
        tfSession=tfSession,
        criticIn=criticIn,
    )

    return estimatedQ

  def updateCritic(self, tfSession: tf.compat.v1.Session, criticUpd: CriticUpdateInput):
    self.critic.updateModel(
        tfSession=tfSession,
        criticUpd=criticUpd,
    )

  def calculateCriticGradients(self, tfSession: tf.compat.v1.Session, inpt: CriticGradientInput):
    gradients = self.critic.calculateGradients(
        tfSession=tfSession,
        inpt=inpt,
    )
    return gradients

  def updateActor(self, tfSession: tf.compat.v1.Session, inpt: ActorUpdateInput):
    self.actor.updateModel(tfSession, inpt)

  def updateTargetModels(self, tfSession: tf.compat.v1.Session):
    self.actorTarget.updateNetParams(tfSession)
    self.criticTarget.updateNetParams(tfSession)
