import tensorflow as tf

from .learning_params import LearningParams
from .critic_dto import CriticEstimateInput, CriticUpdateInput, CriticGradientInput

class CriticMaddpg():
  """ Critic network that estimates the value of the maddpg algorithm"""
  def __init__(self, scope):
    # Number of trainable variables previously declared. Marks the point in which the variables
    # declared by this model reside in the tf.trainable_variables() list
    tfVarBeginIdx = len(tf.trainable_variables())

    # Define the model (input-hidden layers-output)
    self.state = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    self.action = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    self.actionOthers = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    self.inputs = tf.concat([self.state, self.action, self.actionOthers], axis=1)

    # LSTM to encode temporal information
    self.batchSize = tf.placeholder(dtype=tf.int32, shape=[])   # batch size
    self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
    rnnInput = tf.reshape(self.inputs, [self.batchSize, self.trainLength, 3])

    ltsmNumUnits = LearningParams().nnShape.layer_00_ltsm
    ltsmCell = tf.contrib.rnn.BasicLSTMCell(num_units=ltsmNumUnits, state_is_tuple=True)

    self.ltsmInternalState = ltsmCell.zero_state(self.batchSize, tf.float32)
    rnn, self.rnnState = tf.nn.dynamic_rnn(
        inputs=rnnInput,
        cell=ltsmCell,
        dtype=tf.float32,
        initial_state=self.ltsmInternalState,
        scope=scope+'_rnn',
        )
    rnn = tf.reshape(rnn, shape=[-1, ltsmNumUnits])

    # Stack MLP on top of LSTM
    self.Q = CriticMaddpg._buildMlp(rnn) # Critic output is the estimated Q value

    # Params relevant to this network
    self.networkParams = tf.trainable_variables()[tfVarBeginIdx:]

    # Obtained from the target network (double architecture)
    self.targetQ = tf.placeholder(tf.float32,  [None,  1])

    # Loss function and optimization of the critic
    lossFn = tf.reduce_mean(tf.square(self.targetQ-self.Q))
    optimizer = tf.train.AdamOptimizer(1e-4)
    self.updateFn = optimizer.minimize(lossFn)

    # Get the gradient for the actor
    self.criticGradientsFn = tf.gradients(self.Q, self.action)

  @staticmethod
  def _buildMlp(rnn):
    # Perform Xavier initialization of weights
    initializer = tf.contrib.layers.xavier_initializer()

    ltsmNumUnits = LearningParams().nnShape.layer_00_ltsm

    #Layer 1
    l1_units = LearningParams().nnShape.layer_01_mlp_01
    l1_bias = tf.Variable(initializer([1, l1_units]))
    l1_weights = tf.Variable(initializer([ltsmNumUnits, l1_units]))
    layer_1 = tf.nn.relu(tf.matmul(rnn, l1_weights) + l1_bias)

     #Layer 2
    l2_units = LearningParams().nnShape.layer_02_mlp_02
    l2_bias = tf.Variable(initializer([1, l2_units]))
    l2_weights = tf.Variable(initializer([l1_units, l2_units]))
    layer_2 = tf.nn.relu(tf.matmul(layer_1, l2_weights) + l2_bias)

    #Layer 3
    l3_units = LearningParams().nnShape.layer_03_mlp_03
    l3_bias = tf.Variable(initializer([1, l3_units]))
    l3_weights = tf.Variable(initializer([l2_units, l3_units]))
    layer_3 = tf.nn.relu(tf.matmul(layer_2, l3_weights) + l3_bias)

    #Layer 4
    l4_units = LearningParams().nnShape.layer_04_mlp_04
    l4_bias = tf.Variable(initializer([1, l4_units]))
    l4_weights = tf.Variable(initializer([l3_units, l4_units]))
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

  def getEstimatedQ(self, tfSession: tf.Session, criticIn: CriticEstimateInput) -> float:
    estimatedQ = tfSession.run(
        self.Q,
        feed_dict={
            self.state: criticIn.state,
            self.action: criticIn.actionActor,
            self.actionOthers: criticIn.actionsOthers,
            self.trainLength: criticIn.traceLength, ## TODO rename train length
            self.batchSize: criticIn.batchSize,
            self.ltsmInternalState: criticIn.ltsmInternalState,
        }
      )

    return estimatedQ

  def updateModel(self, tfSession: tf.Session, criticUpd: CriticUpdateInput):
    tfSession.run(
        self.updateFn,
        feed_dict={
            self.state: criticUpd.state,
            self.action: criticUpd.actionActor,
            self.actionOthers: criticUpd.actionsOthers,
            self.targetQ: criticUpd.targetQs,
            self.trainLength: criticUpd.traceLength,
            self.batchSize: criticUpd.batchSize,
            self.ltsmInternalState: criticUpd.ltsmInternalState,
        }
      )

  def calculateGradients(self, tfSession: tf.Session, inpt: CriticGradientInput):
    gradients = tfSession.run(
      self.criticGradientsFn,
      feed_dict={
            self.state: inpt.state,
            self.action: inpt.actionActor,
            self.actionOthers: inpt.actionsOthers,
            self.trainLength: inpt.traceLength,
            self.batchSize: inpt.batchSize,
            self.ltsmInternalState: inpt.ltsmInternalState,
        }
    )
    gradients = gradients[0]
    return gradients
