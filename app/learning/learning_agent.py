import tensorflow as tf
import numpy as np

from .maddpg_actor import ActorMaddpg as Actor
from .maddpg_critic import CriticMaddpg as Critic
from .learning_params import LearningParams
from .learning_state import LearningState
from .actor_dto import ActionInput, ActorUpdateInput
from .critic_dto import CriticEstimateInput, CriticUpdateInput, CriticGradientInput

class Agent():
  """Entity representing a single agent in the scenario to be learned. Contains the actors & critics associated with learning"""

  def __init__(self, _id):
    self._id = _id # Unique identifier of the agent
    self.actor       = Actor(scope=f'{_id}_actor')
    self.actorTarget = Actor(scope=f'{_id}_actor_target')
    self.critic       = Critic(scope=f'{_id}_critic')
    self.criticTarget = Critic(scope=f'{_id}_critic_target')

    # Create Op Holders for target networks
    self.actorTarget.createOpHolder(self.actor.networkParams, LearningParams().tau)
    self.criticTarget.createOpHolder(self.critic.networkParams, LearningParams().tau)

    # Initial empty input state (must have size of LTSM)
    ltsmSize = LearningParams().nnShape.layer_00_ltsm
    self.state = (np.zeros([1, ltsmSize]), np.zeros([1, ltsmSize]))

  def getId(self):
    return self._id

  def runActorAction(self, tfSession: tf.Session, currentDeltaF):
    (action, nextState) = self.actor.getAction(
        tfSession=tfSession,
        actionIn=ActionInput(
            actorInput=[[currentDeltaF]],
            ltsmInternalState=self.state,
            batchSize=1,
            traceLength=1,
        )
    )

    self.state = nextState

    return action

  def predictActorAction(self, tfSession: tf.Session, currentDeltaF, ltsmState):
    (action, nextState) = self.actor.getAction(
        tfSession=tfSession,
        actionIn=ActionInput(
            actorInput=currentDeltaF,
            ltsmInternalState=ltsmState,
            batchSize=LearningParams().batchSize,
            traceLength=LearningParams().traceSize,
        )
    )

    return action

  def getActorTargetAction(self, tfSession: tf.Session, state, ltsmState):
    (action, nextState) = self.actor.getAction(
        tfSession=tfSession,
        actionIn=ActionInput(
            actorInput=state,
            ltsmInternalState=ltsmState,
            batchSize=LearningParams().batchSize,
            traceLength=LearningParams().traceSize,
        )
    )

    return action, nextState

  def getTargetCriticEstimatedQ(self, tfSession: tf.Session, criticIn: CriticEstimateInput):
    estimatedQ = self.criticTarget.getEstimatedQ(
        tfSession=tfSession,
        criticIn=criticIn,
    )

    return estimatedQ

  def updateCritic(self, tfSession: tf.Session, criticUpd: CriticUpdateInput):
    self.critic.updateModel(
        tfSession=tfSession,
        criticUpd=criticUpd,
    )

  def calculateCriticGradients(self, tfSession: tf.Session, inpt: CriticGradientInput):
    gradients = self.critic.calculateGradients(
        tfSession=tfSession,
        inpt=inpt,
    )
    return gradients

  def updateActor(self, tfSession: tf.Session, inpt: ActorUpdateInput):
    self.actor.updateModel(tfSession, inpt)
