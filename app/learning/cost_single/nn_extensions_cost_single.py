import tensorflow as tf

from ..actor import Actor
from ..critic import Critic

class NnExtensionCost():
  def _declareStateTensors(self):
    self.genOutput = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32, name='gen_output')
    self.totalOutput = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32, name='total_output')

    return [self.genOutput, self.totalOutput]

  def _unravelStateToFeedDict(self, state):
    '''Transform state list into feed dictionary including all individual tensors for each state component'''
    # State is wrapped much in the same way as other inputs, inside nested arrays
    ## TODO simplify understanding of unravelling
    genOutput = [[s[0]['genOutput']]  for s in state]
    totalOutput = [[s[0]['totalOutput']] for s in state]

    partialFeedDict = {
      self.genOutput: genOutput,
      self.totalOutput: totalOutput,
    }

    return partialFeedDict

# Dual inheritance ensures both actor and critic process the state the exact same way
# Order of inheritance matters to ensure the newly minted class satisfies the Abstract conditions
class ActorCostSingle(NnExtensionCost, Actor):
  pass

class ActorCostSingle(NnExtensionCost, Critic):
  pass
