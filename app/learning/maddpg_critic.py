import tensorflow as tf

class critic_maddpg():
  """ Critic network that estimates the policy of the maddpg algorithm"""
  def __init__(self,h_size,cell,sc,num_variables):

    # Define the model (input-hidden layers-output)
    self.s = tf.placeholder(shape=[None,1],dtype=tf.float32)
    self.a = tf.placeholder(shape=[None,1],dtype=tf.float32)
    self.a_o = tf.placeholder(shape=[None,1],dtype=tf.float32)
    self.inputs = tf.concat([self.s,self.a,self.a_o],axis=1)
    self.initializer = tf.contrib.layers.xavier_initializer()

    # LSTM to encode temporal information
    self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])   # batch size
    self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
    self.rnnInp = tf.reshape(self.inputs,[self.batch_size,self.trainLength,3])

    self.state_in = cell.zero_state(self.batch_size,tf.float32)
    self.rnn,self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnnInp,cell=cell,
                              dtype=tf.float32,initial_state=self.state_in,scope=sc+'_rnn')
    self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])

    # MLP on top of LSTM
    self.b1 = tf.Variable(self.initializer([1,1000]))
    self.W1 = tf.Variable(self.initializer([h_size,1000]))
    self.h1 = tf.nn.relu(tf.matmul(self.rnn,self.W1)+self.b1)

    self.b2 = tf.Variable(self.initializer([1,100]))
    self.W2 = tf.Variable(self.initializer([1000,100]))
    self.h2 = tf.nn.relu(tf.matmul(self.h1,self.W2)+self.b2)

    self.b3 = tf.Variable(self.initializer([1,50]))
    self.W3 = tf.Variable(self.initializer([100,50]))
    self.h3 = tf.nn.relu(tf.matmul(self.h2,self.W3)+self.b3)

    self.b4 = tf.Variable(self.initializer([1,1]))
    self.W4 = tf.Variable(self.initializer([50,1]))
    self.Q = tf.matmul(self.h3,self.W4)+self.b4

    # Take params of the main actor network
    self.network_params = tf.trainable_variables()[num_variables:]

    # Obtained from the target network (double architecture)
    self.target_Q = tf.placeholder(tf.float32, [None, 1])

    # Loss function and optimization of the critic
    self.loss = tf.reduce_mean(tf.square(self.target_Q-self.Q))
    self.optimizer = tf.train.AdamOptimizer(1e-4)
    self.upd = self.optimizer.minimize(self.loss)

    # Get the gradient for the actor
    self.critic_gradients = tf.gradients(self.Q,self.a)

  def createOpHolder(self,params,tau):
    """ Use target network op holder if needed"""
    self.update_network_params = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                  tf.multiply(self.network_params[i], 1. - tau))
                                  for i in range(len(self.network_params))]
