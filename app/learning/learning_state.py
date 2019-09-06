from dataclasses import dataclass
from typing import List
from singleton_decorator import singleton

from .experience_buffer import ExperienceBuffer, LearningExperience
from .epsilon import Epsilon

@dataclass
class EpisodeState:
  cummReward: float
  experiences: List[LearningExperience]
  electricalSystem: any
  allRewards: List[any]

@dataclass
class ModelState:
  allAgents: List[any]
  xpBuffer: ExperienceBuffer
  cummRewardList: List[float]
  epsilon: Epsilon
  allRewards: List[any]

@singleton
@dataclass
class LearningState:
  episode: EpisodeState = None
  model: ModelState = None

  def initData(self, allAgents, xpBuffer, epsilon):
    self.episode = EpisodeState(
        cummReward=0,
        experiences=[],
        electricalSystem=None,
        allRewards=[],
    )
    self.model = ModelState(
        allAgents=allAgents,
        xpBuffer=xpBuffer,
        cummRewardList=[],
        epsilon=epsilon,
        allRewards=[],
    )
