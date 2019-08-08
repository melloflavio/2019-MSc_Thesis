import tensorflow as tf

from .actor_dto import ActionInput, ActionOutput
from .learning_params import LearningParams

_BATCH_SIZE = 32

class ActorMaddpg():
  """ Actor network that estimates the policy of the maddpg algorithm"""
  def __init__(self, scope):
    # Number of trainable variables previously declared. Marks the point in which the variables
    # declared by this model reside in the tf.trainable_variables() list
    tfVarBeginIdx = len(tf.trainable_variables())


    # Define the model (input-hidden layers-output)
    self.inputs = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # LSTM to encode temporal information
    self.batchSize = tf.placeholder(dtype=tf.int32, shape=[])   # batch size
    self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
    rnnInputs = tf.reshape(self.inputs, [self.batchSize, self.trainLength, 1])

    ltsmNumUnits = LearningParams().nnShape.layer_00_ltsm
    ltsmCell = tf.contrib.rnn.BasicLSTMCell(num_units=ltsmNumUnits, state_is_tuple=True)

    self.ltsmInternalState = ltsmCell.zero_state(self.batchSize, tf.float32)
    rnn, self.rnnState = tf.nn.dynamic_rnn(
        inputs=rnnInputs,
        cell=ltsmCell,
        dtype=tf.float32,
        initial_state=self.ltsmInternalState,
        scope=scope+'_rnn',
        )
    rnn = tf.reshape(rnn, shape=[-1, ltsmNumUnits])

    # Stack on top of LSTM
    self.action = ActorMaddpg._buildMlp(rnn)

    # Params relevant to this network
    self.networkParams = tf.trainable_variables()[tfVarBeginIdx:]

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

  def getAction(self, tfSession: tf.Session, actionIn: ActionInput) -> ActionOutput:
    action, nextState = tfSession.run(
        [self.action, self.rnnState],
        feed_dict={
            self.inputs: actionIn.actorInput,
            self.ltsmInternalState: actionIn.ltsmInternalState,
            self.batchSize: actionIn.batchSize,
            self.trainLength: actionIn.traceLength,
        }
      )

    return (action, nextState)

  def getActionOnly(self, tfSession: tf.Session, actionIn: ActionInput) -> ActionOutput:
    action, nextState = tfSession.run(
        [self.action, self.rnnState],
        feed_dict={
            self.inputs: actionIn.actorInput,
            self.ltsmInternalState: actionIn.ltsmInternalState,
            self.batchSize: actionIn.batchSize,
            self.trainLength: actionIn.traceLength,
        }
      )

    return (action, nextState)
