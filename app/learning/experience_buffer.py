from typing import List, Dict, NamedTuple
import numpy as np
import pandas as pd

from .experience_buffer_dto import XpMiniBatch

class LearningExperience(NamedTuple):
  originalState: Dict[str, float]
  destinationState: Dict[str, float]
  actions: Dict[str, float]
  reward: float

class ExperienceBuffer():
  """ Create a buffer to store information to train recurrent models"""
  def __init__(self, bufferSize=100):
    # Buffer is a list of episodes which, in turn, are a list of experiences/steps
    self.buffer: List[List[LearningExperience]] = []
    self.bufferSize = bufferSize

  @property
  def numStoredEpisodes(self):
    return len(self.buffer)

  def add(self, episode: List[LearningExperience]):
    if len(self.buffer) + 1 >= self.bufferSize:
      self.buffer[0:(1+len(self.buffer))-self.bufferSize] = []
    self.buffer.append(episode)

  def getSample(self, batchSize, traceLength):
    index = np.random.choice(np.arange(len(self.buffer)), batchSize)
    sampledEpisodes = [self.buffer[i] for i in index]
    sampledTraces = [
      self._getTraceFromEpisode(episode, traceLength) for episode in sampledEpisodes
    ] # All traces are flattened into a single list

    return self._formatSampleTraces(sampledTraces)

  def _getTraceFromEpisode(self, episode, traceLength) -> List[List[LearningExperience]]:
    initialStep = np.random.randint(0, len(episode)+1-traceLength) # The first step of the trace memory
    traceSteps = episode[initialStep:initialStep+traceLength]
    return traceSteps

  def _formatSampleTraces(self, sampledTraces: List[List[LearningExperience]]):
    originalStates: List[float] = self.consolidateDicts([xp.originalState for trace in sampledTraces for xp in trace])
    destinationStates: List[float] = self.consolidateDicts([xp.destinationState for trace in sampledTraces for xp in trace])

    rewards: List[float] = [[xp.reward] for trace in sampledTraces for xp in trace]

    allActions = self.consolidateDicts([xp.actions for trace in sampledTraces for xp in trace]) # Extract all actions from the sampled traces

    xpMiniBatch = XpMiniBatch(
        originalStates=originalStates,
        destinationStates=destinationStates,
        allActions=allActions,
        rewards=rewards,
      )
    return xpMiniBatch

  def consolidateDicts(self, dictList: List[Dict[str, float]]):
    '''Consolidates a list of dictionaries into a single dictionary with lists for each key'''
    # Turns [{a:1, b:2}, {a:3, b:4}, {a:5, b:6}]
    # into {a:[1, 3, 5], b:[2, 4, 6]}
    df = pd.DataFrame(dictList)
    df = df.applymap(lambda x: [x]) # Wrap individual values in arrays
    return df.to_dict(orient='list')
