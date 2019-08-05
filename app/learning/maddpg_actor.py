import tensorflow as tf

_BATCH_SIZE = 32

class ActorMaddpg():
  """ Actor network that estimates the policy of the maddpg algorithm"""
  def __init__(self, hSize, cell, scope, numVariables):

    # Define the model (input-hidden layers-output)
    self.inputs = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # LSTM to encode temporal information
    self.batchSize = tf.placeholder(dtype=tf.int32, shape=[])   # batch size
    self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
    rnnInputs = tf.reshape(self.inputs, [self.batchSize, self.trainLength, 1])

    self.stateIn = cell.zero_state(self.batchSize, tf.float32)
    rnn, self.rnnState = tf.nn.dynamic_rnn(
        inputs=rnnInputs,
        cell=cell,
        dtype=tf.float32,
        initial_state=self.stateIn,
        scope=scope+'_rnn',
        )
    rnn = tf.reshape(rnn, shape=[-1, hSize])

    # Stack on top of LSTM
    self.action = ActorMaddpg._buildMlp(rnn, hSize)

    # Take params of the main actor network
    self.networkParams = tf.trainable_variables()[numVariables:]

    # This gradient will be provided by the critic network
    self.criticGradient = tf.placeholder(tf.float32, [None, 1])

    # Take the gradients and combine
    unnormalizedActorGradients = tf.gradients(
                self.action, self.networkParams, -self.criticGradient)

    # Normalize dividing by the size of the batch (gradients sum all over the batch)
    self.actorGradients = list(map(lambda x: tf.div(x, _BATCH_SIZE), unnormalizedActorGradients))

    # Optimization of the actor
    self.optimizer = tf.train.AdamOptimizer(1e-4)
    self.upd = self.optimizer.apply_gradients(zip(self.actorGradients, self.networkParams))

  @staticmethod
  def _buildMlp(rnn, hSize):
    # Perform Xavier initialization of weights
    initializer = tf.contrib.layers.xavier_initializer()

    #Layer 1
    l1_bias = tf.Variable(initializer([1, 1000]))
    l1_weights = tf.Variable(initializer([hSize, 1000]))
    layer_1 = tf.nn.relu(tf.matmul(rnn, l1_weights) + l1_bias)

    #Layer 2
    l2_bias = tf.Variable(initializer([1, 100]))
    l2_weights = tf.Variable(initializer([1000, 100]))
    layer_2 = tf.nn.relu(tf.matmul(layer_1, l2_weights) + l2_bias)

    #Layer 3
    l3_bias = tf.Variable(initializer([1, 50]))
    l3_weights = tf.Variable(initializer([100, 50]))
    layer_3 = tf.nn.relu(tf.matmul(layer_2, l3_weights) + l3_bias)

    #Layer 4
    l4_bias = tf.Variable(initializer([1, 1]))
    l4_weights = tf.Variable(initializer([50, 1]))
    actionUnscaled = tf.nn.tanh(tf.matmul(layer_3, l4_weights) + l4_bias)

    action = tf.multiply(actionUnscaled, 0.1)

    return action

  def createOpHolder(self, params, tau):
    """ Use target network op holder if needed"""
    networkParamSize = len(self.networkParams)
    self.updateNetworkParams = [None]*networkParamSize

    for i in range(networkParamSize):
      assignAction = self.networkParams[i].assign(
          tf.multiply(params[i], tau) + tf.multiply(self.networkParams[i], 1. - tau))
      self.updateNetworkParams[i] = assignAction
