def costRewardFunction(totalOutputTarget, totalOutputDestination, totalCost):
    totalCost = totalCost/(1000000.0) # Scale down cost to levels near the ones found in output differential (e.g. 0.1 */ 10)
    outputDifferential = (totalOutputDestination - totalOutputTarget)/totalOutputTarget
    earnedReward = 1/(2**(200*(totalCost**2) + (outputDifferential**2)*500))
    return earnedReward
