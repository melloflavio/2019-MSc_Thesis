import tensorflow as tf

class CriticMaddpg():
  """ Critic network that estimates the value of the maddpg algorithm"""
  def __init__(self, hSize, scope, numVariables):

    # Define the model (input-hidden layers-output)
    self.state = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    self.action = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    self.actionOther = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    self.inputs = tf.concat([self.state, self.action, self.actionOther], axis=1)

    # LSTM to encode temporal information
    self.batchSize = tf.placeholder(dtype=tf.int32, shape=[])   # batch size
    self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
    rnnInput = tf.reshape(self.inputs, [self.batchSize, self.trainLength, 3])

    ltsmCell = tf.contrib.rnn.BasicLSTMCell(num_units=hSize, state_is_tuple=True)

    self.stateIn = ltsmCell.zero_state(self.batchSize, tf.float32)
    rnn, self.rnnState = tf.nn.dynamic_rnn(
        inputs=rnnInput,
        cell=ltsmCell,
        dtype=tf.float32,
        initial_state=self.stateIn,
        scope=scope+'_rnn',
        )
    rnn = tf.reshape(rnn, shape=[-1, hSize])

    # Stack MLP on top of LSTM
    self.Q = CriticMaddpg._buildMlp(rnn, hSize) # Critic output is the estimated Q value

    # Take params of the main actor network
    self.networkParams = tf.trainable_variables()[numVariables:]

    # Obtained from the target network (double architecture)
    self.targetQ = tf.placeholder(tf.float32,  [None,  1])

    # Loss function and optimization of the critic
    lossFn = tf.reduce_mean(tf.square(self.targetQ-self.Q))
    optimizer = tf.train.AdamOptimizer(1e-4)
    self.upd = optimizer.minimize(lossFn)

    # Get the gradient for the actor
    self.critic_gradients = tf.gradients(self.Q, self.action)

  @staticmethod
  def _buildMlp(rnn, hSize):
    # Perform Xavier initialization of weights
    initializer = tf.contrib.layers.xavier_initializer()

    #Layer 1
    l1_bias = tf.Variable(initializer([1, 1000]))
    l1_weights = tf.Variable(initializer([hSize, 1000]))
    layer_1 = tf.nn.relu(tf.matmul(rnn, l1_weights)+l1_bias)

    #Layer 2
    l2_bias = tf.Variable(initializer([1, 100]))
    l2_weights = tf.Variable(initializer([1000, 100]))
    layer_2 = tf.nn.relu(tf.matmul(layer_1, l2_weights)+l2_bias)

    #Layer 3
    l3_bias = tf.Variable(initializer([1, 50]))
    l3_weights = tf.Variable(initializer([100, 50]))
    layer_3 = tf.nn.relu(tf.matmul(layer_2, l3_weights)+l3_bias)

    #Layer 4
    l4_bias = tf.Variable(initializer([1, 1]))
    l4_weights = tf.Variable(initializer([50, 1]))
    qValue = tf.matmul(layer_3, l4_weights)+l4_bias # Critic output is the estimated Q value

    return qValue

  def createOpHolder(self, params, tau):
    """ Use target network op holder if needed"""
    networkParamSize = len(self.networkParams)
    self.updateNetworkParams = [None]*networkParamSize

    for i in range(networkParamSize):
      assignAction = self.networkParams[i].assign(
          tf.multiply(params[i],  tau) + tf.multiply(self.networkParams[i],  1. - tau))
      self.updateNetworkParams[i] = assignAction
