import numpy as np

class experience_buffer():
  """ Create a buffer to store information to train recurrent models"""
  def __init__(self, buffer_size = 100):
    self.buffer = []
    self.buffer_size = buffer_size

  def add(self,experience):
    if len(self.buffer) + 1 >= self.buffer_size:
      self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
    self.buffer.append(experience)

  def sample(self,batch_size,trace_length,n_var):
    index = np.random.choice(np.arange(len(self.buffer)),batch_size)
    sampled_episodes = [self.buffer[i] for i in index]
    sampledTraces = []
    for episode in sampled_episodes:
      point = np.random.randint(0,episode.shape[0]+1-trace_length)
      sampledTraces.append(episode[point:point+trace_length,:])
    return np.reshape(np.array(sampledTraces),[-1,n_var])
