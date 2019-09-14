import tensorflow as tf

from ..actor import Actor
from ..critic import Critic

class NnExtensionFrequency():
  def _declareStateTensors(self):
    self.deltaFreq = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32, name='delta_freq')

    return [self.deltaFreq]

  def _unravelStateToFeedDict(self, state):
    '''Transform state list into feed dictionary including all individual tensors for each state component'''
    # State is wrapped much in the same way as other inputs, inside nested arrays
    ## TODO simplify understanding of unravelling
    deltaFreq = [[s[0]['deltaFreq']] for s in state]

    partialFeedDict = {
      self.deltaFreq: deltaFreq,
    }

    return partialFeedDict

# Dual inheritance ensures both actor and critic process the state the exact same way
# Order of inheritance matters to ensure the newly minted class satisfies the Abstract conditions
class ActorFrequency(NnExtensionCostFrequency, Actor):
  pass

class CriticFrequency(NnExtensionCostFrequency, Critic):
  pass
