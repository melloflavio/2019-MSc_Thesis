from enum import Enum, unique

class REWARD_STRATEGY(Enum):
  FREQUENCY = 1,
  COST = 2,

class RewardCalculator():

  @staticmethod
  def calculateReward(strategy: REWARD_STRATEGY, deltaFreq: float, deltaCost: float):
    if strategy is REWARD_STRATEGY.FREQUENCY:
      return RewardCalculator._power(deltaFreq)
    elif strategy is REWARD_STRATEGY.COST:
      return RewardCalculator._power(deltaCost)
    else:
      return 0

  @staticmethod
  def _power(value: float):
    power = 2**(10-abs(value))
    return power
