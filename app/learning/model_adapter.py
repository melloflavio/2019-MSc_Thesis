from abc import ABC, abstractmethod

from .actor import Actor
from .critic import Critic

class ModelAdapter(ABC):

  def __init__(self, rewardFn=None):
    # Bind rewardFn property to class method https://stackoverflow.com/a/41921291
    self._rewardFn = rewardFn if rewardFn is not None else self._defaultRewardFunction.__func__

  @property
  @abstractmethod
  def SCOPE_PREFIX(self) -> str:
    pass

  @property
  @abstractmethod
  def Actor(self) -> Actor:
    pass

  @property
  @abstractmethod
  def Critic(self) -> Critic:
    pass

  @classmethod
  @abstractmethod
  def _defaultRewardFunction(**args):
    pass

  @abstractmethod
  def shouldStopEarly(self, elecSystem):
    pass

  @abstractmethod
  def storeInitialState(self, elecSystem, allAgents):
    """Stores initial state. May be used for reward and future state observations"""
    pass

  @abstractmethod
  def observeStates(self, elecSystem, allAgents):
    pass

  @abstractmethod
  def storePreActionStateReward(self, elecSystem):
    '''Stores state values which may be used later to calcluate reward. Called before system executes actions.'''
    pass

  @abstractmethod
  def calculateReward(self, elecSystem):
    pass
