from typing import NamedTuple, List
from singleton_decorator import singleton

from .experience_buffer import ExperienceBuffer, LearningExperience

class EpisodeState(NamedTuple):
  cummReward: float
  experiences: List[LearningExperience]

class ModelState(NamedTuple):
  allAgents: List[any]
  xpBuffer: ExperienceBuffer
  cummRewardList: List[float]

@singleton
class LearningState(NamedTuple):
  episode: EpisodeState
  model: ModelState


  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.resetEpisode()

  def resetEpisode(self):
    self.episode.cummReward = 0
