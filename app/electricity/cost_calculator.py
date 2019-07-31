from typing import List
import scipy.optimize as opt

from dto import CostProfile, NodeStateCost

class CostCalculator:

  # Calculates the cost to output a given power with a given generator cost profile
  @staticmethod
  def calculateCost(power, costProfile: CostProfile) -> float:
    cost = costProfile.alpha + costProfile.beta*power + costProfile.gamma*(power**2)
    return cost

  # Generates an objective function which calculates the total cost given a power output combination
  @staticmethod
  def _getObjectiveFn(generators):
    def objectiveFn(powers):
      print(powers)
      generatorPowerTuples = zip(powers, generators)
      totalCost = sum([
          CostCalculator.calculateCost(power, generator.getCostProfile())
          for (power, generator) in generatorPowerTuples
          ])
      return totalCost
    return objectiveFn

  # Generates all constraints associated with the optimization problem
  # - Total sum must match given total power
  # - Power outputs must be positive
  # - Power outputs must be within each generator's limits
  # Note: SciPy inequalities are in the from <computed_value> >=0
  @staticmethod
  def _generateConstraints(allGenerators: List[any], totalPower):
    generatorIndexes = range(len(allGenerators))

    constraints = []

    # Total power output must match the load (or the total power currently injected by the generators)
    constraints.append({'type': 'eq', 'fun': lambda x: sum(x) - totalPower})

    # Every generator must have output >= 0 (sanity check)
    constraints.extend([{'type': 'ineq', 'fun': lambda x: x[i]} for i in generatorIndexes])

    # Outputs must respect each generators maximum capacity
    constraints.extend([{
          'type': 'ineq',
          'fun': lambda x: - x[idx] + generator.getMaxPower()
        } for idx, generator in enumerate(allGenerators)
      ])

    # Outputs must respect each generators minimum capacity
    constraints.extend([{
          'type': 'ineq',
          'fun': lambda x: x[idx] - generator.getMinPower()
        } for idx, generator in enumerate(allGenerators)
      ])

    return constraints


  @staticmethod
  def calculateMinimumCost(allGenerators, totalPower):
    # Generate the objective function using the generators
    objective = CostCalculator._getObjectiveFn(allGenerators)

    # Start optimization with current power setup
    initialGuess = [generator.getOutput() for generator in allGenerators]

    # Generate all relevant contraints
    constraints = CostCalculator._generateConstraints(allGenerators, totalPower)

    results = opt.minimize(
        objective,
        initialGuess,
        constraints=constraints,
        # options={'maxiter':20, 'disp':True}
        )

    minCost = results.fun
    minSetup = zip(allGenerators, results.x) # Assigned power for each generator
    minCostNodes = [NodeStateCost(id_=generator.getId(), cost=cost) for (generator, cost) in minSetup]

    return minCost, minCostNodes
