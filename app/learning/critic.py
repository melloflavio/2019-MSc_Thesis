import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod

from .learning_params import LearningParams
from .critic_dto import CriticEstimateInput, CriticUpdateInput, CriticGradientInput

class Critic(ABC):
  """ Critic network that estimates the value of the maddpg algorithm"""
  def __init__(self, scope):
    # Number of trainable variables previously declared. Marks the point in which the variables
    # declared by this model reside in the tf.trainable_variables() list
    tfVarBeginIdx = len(tf.compat.v1.trainable_variables())

    with tf.name_scope(scope):

      # Define the model (input-hidden layers-output)
      stateTensors = self._declareStateTensors()
      self.action = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32, name='action') # action taken by actor of the same agent
      self.actionOthers = self.declareActionOthersTensors()
      self.inputs = tf.concat([*stateTensors, self.action, *self.actionOthers], axis=1)

      # LSTM to encode temporal information
      numInputVars = self.inputs.get_shape()[1]
      self.batchSize = tf.compat.v1.placeholder(dtype=tf.int32, shape=[], name='batch_size')   # batch size
      self.traceLength = tf.compat.v1.placeholder(dtype=tf.int32, name='trace_length')           # trace lentgth
      rnnInput = tf.reshape(self.inputs, [self.batchSize, self.traceLength, numInputVars])

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
      self.Q = self._buildMlp(rnn) # Critic output is the estimated Q value

      # Params relevant to this network
      self.networkParams = tf.compat.v1.trainable_variables()[tfVarBeginIdx:]

      # Obtained from the target network (double architecture)
      self.targetQ = tf.compat.v1.placeholder(tf.float32,  [None,  1], name='target_q')

      # Loss function and optimization of the critic
      lossFn = tf.reduce_mean(tf.square(self.targetQ-self.Q))
      optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
      self.updateFn = optimizer.minimize(lossFn)

      # Get the gradient for the actor
      self.criticGradientsFn = tf.gradients(self.Q, self.action)

  def _buildMlp(self, rnn):
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

  def getEstimatedQ(self, tfSession: tf.compat.v1.Session, criticIn: CriticEstimateInput, ltsmState) -> float:
    # Unravel state into individual components
    stateDict = self._unravelStateToFeedDict(criticIn.state)
    actionsOthersDict = self.unravelActionOthersInputToFeedDict(criticIn)

    estimatedQ = tfSession.run(
        self.Q,
        feed_dict={
            **stateDict,
            **actionsOthersDict,
            self.action: criticIn.actionActor,
            self.traceLength: criticIn.traceLength,
            self.batchSize: criticIn.batchSize,
            self.ltsmInternalState: ltsmState,
        }
      )

    return estimatedQ

  def updateModel(self, tfSession: tf.compat.v1.Session, criticUpd: CriticUpdateInput, ltsmState):
    # Unravel state into individual components
    stateDict = self._unravelStateToFeedDict(criticUpd.state)
    actionsOthersDict = self.unravelActionOthersInputToFeedDict(criticUpd)

    tfSession.run(
        self.updateFn,
        feed_dict={
            **stateDict,
            **actionsOthersDict,
            self.action: criticUpd.actionActor,
            self.targetQ: criticUpd.targetQs,
            self.traceLength: criticUpd.traceLength,
            self.batchSize: criticUpd.batchSize,
            self.ltsmInternalState: ltsmState,
        }
      )

  def calculateGradients(self, tfSession: tf.compat.v1.Session, inpt: CriticGradientInput, ltsmState):

    stateDict = self._unravelStateToFeedDict(inpt.state)

    actionsOthersDict = self.unravelActionOthersInputToFeedDict(inpt)

    gradients = tfSession.run(
      self.criticGradientsFn,
      feed_dict={
            **stateDict,
            **actionsOthersDict,
            self.action: inpt.actionActor,
            self.traceLength: inpt.traceLength,
            self.batchSize: inpt.batchSize,
            self.ltsmInternalState: ltsmState,
        }
    )
    gradients = gradients[0]
    return gradients

  def updateNetParams(self, tfSession: tf.compat.v1.Session):
    tfSession.run(self.updateNetworkParams)

  def declareActionOthersTensors(self):
    numAgents = len(LearningParams().electricalSystemSpecs.generators)

    actionOthers = []
    for i in range(numAgents - 1): # -1 since this agent is one of the list
      tensorName = f'actions_others_{i}'
      actionOtherTensor = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32, name=tensorName) # actions taken by actors of other agents
      actionOthers.append(actionOtherTensor)
    return actionOthers

  def unravelActionOthersInputToFeedDict(self, inpt):
    numAgents = len(LearningParams().electricalSystemSpecs.generators)

    actionOthersValues = inpt.actionsOthers
    batchSize = inpt.batchSize
    traceLength = inpt.traceLength

    reshapedActions = np.reshape(actionOthersValues, (-1, batchSize * traceLength))

    actionsOthersDict = {
      tensor: [[a] for a in actions]
      for tensor, actions in zip(self.actionOthers, reshapedActions)
    }

    return actionsOthersDict

  #############################
  # Begin Abstract Methods
  #############################

  @abstractmethod
  def _declareStateTensors(self):
    pass

  @abstractmethod
  def _unravelStateToFeedDict(self, state):
    '''Transform state list into feed dictionary including all individual tensors for each state component'''
    pass
