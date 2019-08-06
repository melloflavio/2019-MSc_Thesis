import tensorflow as tf

from .maddpg_actor import ActorMaddpg as Actor
from .maddpg_critic import CriticMaddpg as Critic

class Agent():
  """Entity representing a single agent in the scenario to be learned. Contains the actors & critics associated with learning"""

  def __init__(self, _id):
    self._id = _id # Unique identifier of the agent
    self.actor = Actor(scope=f'{_id}_actor', numVariables=0)
    self.critic = Critic(scope=f'{_id}_critic', numVariables=len(tf.trainable_variables()))
    self.actorTarget = Actor(scope=f'{_id}_actor_target', numVariables=len(tf.trainable_variables()))
    self.criticTarget = Critic(scope=f'{_id}_critic_target', numVariables=len(tf.trainable_variables()))
