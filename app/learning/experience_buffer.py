import numpy as np

class ExperienceBuffer():
  """ Create a buffer to store information to train recurrent models"""
  def __init__(self, bufferSize=100):
    self.buffer = []
    self.bufferSize = bufferSize

  def add(self, experience):
    if len(self.buffer) + 1 >= self.bufferSize:
      self.buffer[0:(1+len(self.buffer))-self.bufferSize] = []
    self.buffer.append(experience)

  def sample(self, batchSize, traceLength, numVars):
    index = np.random.choice(np.arange(len(self.buffer)), batchSize)
    sampledEpisodes = [self.buffer[i] for i in index]
    sampledTraces = []
    for episode in sampledEpisodes:
      point = np.random.randint(0, episode.shape[0]+1-traceLength)
      sampledTraces.append(episode[point:point+traceLength, :])
    return np.reshape(np.array(sampledTraces), [-1, numVars])
