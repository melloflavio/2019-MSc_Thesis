from typing import NamedTuple, List
from singleton_decorator import singleton

@singleton
class LearningState(NamedTuple):
  cummReward: float
  cummRewardList: List[float]

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.reset()

  def reset(self):
    self.cummReward = 0
    self.cummRewardList = []
