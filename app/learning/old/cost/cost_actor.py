import tensorflow as tf

from ..actor_dto import ActionInput, ActionOutput, ActorUpdateInput
from ..learning_params import LearningParams

_BATCH_SIZE = 32

class CostActorMaddpg():
  """ Actor network that estimates the policy of the maddpg algorithm"""
  def __init__(self, scope):
    # Number of trainable variables previously declared. Marks the point in which the variables
    # declared by this model reside in the tf.trainable_variables() list
    tfVarBeginIdx = len(tf.compat.v1.trainable_variables())

    with tf.name_scope(scope):

      # Define the model (input-hidden layers-output)
      # self.inputs = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32, name='inputs')
      self.genOutput = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32, name='gen_output')
      self.totalOutput = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32, name='total_output')
      self.inputs = tf.concat([self.genOutput, self.totalOutput], axis=1)

      # LSTM to encode temporal information
      numInputVars = self.inputs.get_shape()[1]
      self.batchSize = tf.compat.v1.placeholder(dtype=tf.int32, shape=[], name='batch_size')   # batch size
      self.traceLength = tf.compat.v1.placeholder(dtype=tf.int32, name='trace_length')           # trace lentgth
      rnnInputs = tf.reshape(self.inputs, [self.batchSize, self.traceLength, numInputVars])

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
      self.action = CostActorMaddpg._buildMlp(rnn)

      # Params relevant to this network
      self.networkParams = tf.compat.v1.trainable_variables()[tfVarBeginIdx:]

      # This gradient will be provided by the critic network
      self.criticGradient = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1], name='critic_gradient')

      # Take the gradients and combine
      unnormalizedActorGradients = tf.gradients(
                  self.action, self.networkParams, -self.criticGradient)

      # Normalize dividing by the size of the batch (gradients sum all over the batch)
      self.actorGradients = list(map(lambda x: tf.divide(x, _BATCH_SIZE), unnormalizedActorGradients))

      # Optimization of the actor
      self.optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
      self.upd = self.optimizer.apply_gradients(zip(self.actorGradients, self.networkParams))

  @staticmethod
  def _buildMlp(rnn):
    # Perform Xavier initialization of weights
    initializer = tf.contrib.layers.xavier_initializer()

    ltsmNumUnits = LearningParams().nnShape.layer_00_ltsm

    #Layer 1
    l1_units = LearningParams().nnShape.layer_01_mlp_01
    l1_bias = tf.Variable(initializer([1, l1_units]), name='l1_bias')
    l1_weights = tf.Variable(initializer([ltsmNumUnits, l1_units]), name='l1_weights')
    layer_1 = tf.nn.relu(tf.matmul(rnn, l1_weights) + l1_bias)

    #Layer 2
    l2_units = LearningParams().nnShape.layer_02_mlp_02
    l2_bias = tf.Variable(initializer([1, l2_units]), name='l2_bias')
    l2_weights = tf.Variable(initializer([l1_units, l2_units]), name='l2_weights')
    layer_2 = tf.nn.relu(tf.matmul(layer_1, l2_weights) + l2_bias)

    #Layer 3
    l3_units = LearningParams().nnShape.layer_03_mlp_03
    l3_bias = tf.Variable(initializer([1, l3_units]), name='l3_bias')
    l3_weights = tf.Variable(initializer([l2_units, l3_units]), name='l3_weights')
    layer_3 = tf.nn.relu(tf.matmul(layer_2, l3_weights) + l3_bias)

    #Layer 4
    l4_units = LearningParams().nnShape.layer_04_mlp_04
    l4_bias = tf.Variable(initializer([1, l4_units]), name='l4_bias')
    l4_weights = tf.Variable(initializer([l3_units, l4_units]), name='l4_weights')
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

  def getAction(self, tfSession: tf.compat.v1.Session, actionIn: ActionInput) -> ActionOutput:
    # Unravel state into individual components
    genOutput = [[s[0]['genOutput']]  for s in actionIn.actorInput]
    totalOutput = [[s[0]['totalOutput']] for s in actionIn.actorInput]

    action, nextState = tfSession.run(
        [self.action, self.rnnState],
        feed_dict={
            self.genOutput: genOutput,
            self.totalOutput: totalOutput,
            self.ltsmInternalState: actionIn.ltsmInternalState,
            self.batchSize: actionIn.batchSize,
            self.traceLength: actionIn.traceLength,
        }
      )

    return (action, nextState)

  def getActionOnly(self, tfSession: tf.compat.v1.Session, actionIn: ActionInput) -> ActionOutput:
    # Unravel state into individual components
    genOutput = [[s[0]['genOutput']]  for s in actionIn.actorInput]
    totalOutput = [[s[0]['totalOutput']] for s in actionIn.actorInput]

    action = tfSession.run(
        self.action,
        feed_dict={
            self.genOutput: genOutput,
            self.totalOutput: totalOutput,
            self.ltsmInternalState: actionIn.ltsmInternalState,
            self.batchSize: actionIn.batchSize,
            self.traceLength: actionIn.traceLength,
        }
    )

    return action

  def updateModel(self, tfSession: tf.compat.v1.Session, inpt: ActorUpdateInput):
    # Unravel state into individual components
    genOutput = [[s[0]['genOutput']]  for s in inpt.state]
    totalOutput = [[s[0]['totalOutput']] for s in inpt.state]
    tfSession.run(
      self.upd,
      feed_dict={
          self.genOutput: genOutput,
          self.totalOutput: totalOutput,
          self.criticGradient: inpt.gradients,
          self.ltsmInternalState: inpt.ltsmInternalState,
          self.batchSize: inpt.batchSize,
          self.traceLength: inpt.traceLength,
        }
    )

  def updateNetParams(self, tfSession: tf.compat.v1.Session):
    tfSession.run(self.updateNetworkParams)
