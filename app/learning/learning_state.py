from dataclasses import dataclass
from typing import NamedTuple, List
from singleton_decorator import singleton

from .experience_buffer import ExperienceBuffer, LearningExperience

@dataclass
class EpisodeState:
  cummReward: float
  experiences: List[LearningExperience]
  electricalSystem: any

@dataclass
class ModelState:
  allAgents: List[any]
  xpBuffer: ExperienceBuffer
  cummRewardList: List[float]
  epsilon: float
  electricalSystemSpecs: any

@singleton
@dataclass
class LearningState:
  episode: EpisodeState = None
  model: ModelState = None

  def initData(self, allAgents, xpBuffer, epsilon, electricalSystemSpecs):
    self.episode = EpisodeState(
        cummReward=0,
        experiences=[],
        electricalSystem=None,
    )
    self.model = ModelState(
        allAgents=allAgents,
        xpBuffer=xpBuffer,
        cummRewardList=[],
        epsilon=epsilon,
        electricalSystemSpecs=electricalSystemSpecs,
    )
