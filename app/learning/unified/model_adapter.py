from abc import ABC, abstractmethod

class ModelAdapter(ABC):

  def __init__(self, rewardFn=None):
    # Bind rewardFn property to class method https://stackoverflow.com/a/41921291
    self._rewardFn = rewardFn if rewardFn is not None else self._defaultRewardFunction.__func__

  @property
  @abstractmethod
  def SCOPE_PREFIX(self):
    pass

  @property
  @abstractmethod
  def Actor(self):
    pass

  @property
  @abstractmethod
  def Critic(self):
    pass

  @classmethod
  @abstractmethod
  def _defaultRewardFunction(**args):
    pass

  @abstractmethod
  def shouldStopEarly(self, elecSystem):
    pass

  ## TODO enforce typing
  @abstractmethod
  def observeStates(self, elecSystem):
    pass

  @abstractmethod
  def calculateReward(self, elecSystem):
    pass
