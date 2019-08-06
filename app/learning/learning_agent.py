import tensorflow as tf

from .maddpg_actor import ActorMaddpg as Actor
from .maddpg_critic import CriticMaddpg as Critic
from .learning_params import LearningParams
import numpy as np
import tensorflow as tf

class Agent():
  """Entity representing a single agent in the scenario to be learned. Contains the actors & critics associated with learning"""

  def __init__(self, _id):
    self._id = _id # Unique identifier of the agent
    self.actor = Actor(scope=f'{_id}_actor', numVariables=0)
    self.critic = Critic(scope=f'{_id}_critic', numVariables=len(tf.trainable_variables()))
    self.actorTarget = Actor(scope=f'{_id}_actor_target', numVariables=len(tf.trainable_variables()))
    self.criticTarget = Critic(scope=f'{_id}_critic_target', numVariables=len(tf.trainable_variables()))

    # Create Op Holders for target networks
    self.actorTarget.createOpHolder(self.actor.network_params, LearningParams().tau)
    self.criticTarget.createOpHolder(self.critic.network_params, LearningParams().tau)

    # Initial empty input state (must have size of LTSM)
    ltsmSize = LearningParams().nnShape.layer_00_ltsm
    self.state = (np.zeros([1, ltsmSize]), np.zeros([1, ltsmSize]))

  def getId(self):
    return self._id

  def getActorAction(self, tfSession: tf.Session, currentDeltaF):
    action, nextState = tfSession.run(
        [self.actor.action, self.rnnState],
        feed_dict={
            self.actor.inputs: np.array(currentDeltaF).reshape(1,1),
            self.actor.stateIn: self.state,
            self.actor.batchSize:1,
            self.actor.trainLength:1,
        }
      )

    action = action[0,0] + LearningParams().epsilon * np.random.normal(0.0,0.4)
    self.state = nextState

    return action
