from dto import CostProfile

class CostCalculator:

  @staticmethod
  def calculateCost(power, costProfile: CostProfile) -> float:
    cost = costProfile.alpha + costProfile.beta*power + costProfile.gamma*(power**2)
    return cost
