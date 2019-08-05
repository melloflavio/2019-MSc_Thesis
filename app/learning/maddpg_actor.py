import tensorflow as tf

class actor_maddpg():
  """ Actor network that estimates the policy of the maddpg algorithm"""
  def __init__(self, hSize, cell, scope, numVariables):

    # Define the model (input-hidden layers-output)
    self.inputs = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    self.initializer = tf.contrib.layers.xavier_initializer()

    # LSTM to encode temporal information
    self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])   # batch size
    self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
    self.rnnInp = tf.reshape(self.inputs, [self.batch_size, self.trainLength, 1])

    self.state_in = cell.zero_state(self.batch_size, tf.float32)
    self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnnInp, cell=cell,
                              dtype=tf.float32, initial_state=self.state_in, scope=scope+'_rnn')
    self.rnn = tf.reshape(self.rnn, shape=[-1, hSize])

    # MLP on top of LSTM
    self.layer1Bias = tf.Variable(self.initializer([1, 1000]))
    self.layer1Weights = tf.Variable(self.initializer([hSize, 1000]))
    self.layer1 = tf.nn.relu(tf.matmul(self.rnn, self.layer1Weights)+self.layer1Bias)

    self.layer2Bias = tf.Variable(self.initializer([1, 100]))
    self.layer2Weights = tf.Variable(self.initializer([1000, 100]))
    self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.layer2Weights)+self.layer2Bias)

    self.layer3Bias = tf.Variable(self.initializer([1, 50]))
    self.layer3Weights = tf.Variable(self.initializer([100, 50]))
    self.layer3 = tf.nn.relu(tf.matmul(self.layer2, self.layer3Weights)+self.layer3Bias)

    self.layer4Bias = tf.Variable(self.initializer([1, 1]))
    self.layer4Weights = tf.Variable(self.initializer([50, 1]))
    self.actionUnscaled = tf.nn.tanh(tf.matmul(self.layer3, self.layer4Weights)+self.layer4Bias)
    self.action = tf.multiply(self.actionUnscaled, 0.1)

    # Take params of the main actor network
    self.networkParams = tf.trainable_variables()[numVariables:]

    # This gradient will be provided by the critic network
    self.criticGradient = tf.placeholder(tf.float32, [None, 1])

    # Take the gradients and combine
    self.unnormalizedActorGradients = tf.gradients(
                self.action, self.networkParams, -self.criticGradient)

    # Normalize dividing by the size of the batch (gradients sum all over the batch)
    self.actorGradients = list(map(lambda x: tf.div(x, 32),
                                    self.unnormalizedActorGradients))

    # Optimization of the actor
    self.optimizer = tf.train.AdamOptimizer(1e-4)
    self.upd = self.optimizer.apply_gradients(zip(self.actorGradients, self.networkParams))

  def createOpHolder(self, params, tau):
    """ Use target network op holder if needed"""
    self.updateNetworkParams = [self.networkParams[i].assign(tf.multiply(params[i], tau) +
                                  tf.multiply(self.networkParams[i], 1. - tau))
                                  for i in range(len(self.networkParams))]
