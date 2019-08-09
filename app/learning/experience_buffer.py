from typing import List, Dict, NamedTuple
import numpy as np

from .experience_buffer_dto import XpMiniBatch

class LearningExperience(NamedTuple):
  originalState: float
  destinationState: float
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
    originalStates: List[float] = [[xp.originalState] for trace in sampledTraces for xp in trace]
    destinationStates: List[float] = [[xp.destinationState] for trace in sampledTraces for xp in trace]
    rewards: List[float] = [[xp.reward] for trace in sampledTraces for xp in trace]

    # Group actions by agent. i.e. Transform a list of dictionaries into a dictionary of lists
    allActions = [xp.actions for trace in sampledTraces for xp in trace] # Extract all actions from the sampled traces
    allAgentIds = allActions[0].keys()               # Get all agent ids from the first trace found (all steps should have the same actors, otherwise the consolidated actions would be misaligned)
    groupedActions = {agentId: [[action.get(agentId)] for action in allActions] for agentId in allAgentIds} # Create new dict, each key is the agentId, values are the list of actions said agent has taken in the trace

    xpMiniBatch = XpMiniBatch(
        originalStates=originalStates,
        destinationStates=destinationStates,
        groupedActions=groupedActions,
        rewards=rewards,
      )
    return xpMiniBatch
