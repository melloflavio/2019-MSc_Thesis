from dataclasses import dataclass
from typing import NamedTuple, List
from singleton_decorator import singleton

from .experience_buffer import ExperienceBuffer, LearningExperience

@dataclass
class EpisodeState:
  cummReward: float
  experiences: List[LearningExperience]

@dataclass
class ModelState:
  allAgents: List[any]
  xpBuffer: ExperienceBuffer
  cummRewardList: List[float]
  epsilon: float

@singleton
@dataclass
class LearningState:
  episode: EpisodeState = None
  model: ModelState = None

  def initData(self, allAgents, xpBuffer, epsilon):
    self.episode = EpisodeState(
        cummReward=0,
        experiences=[],
    )
    self.model = ModelState(
        allAgents=allAgents,
        xpBuffer=xpBuffer,
        cummRewardList=[],
        epsilon=epsilon,
    )
