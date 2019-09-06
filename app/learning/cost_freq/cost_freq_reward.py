def costRewardFunction(deltaFreq, totalCost):
    totalCost = totalCost/(10000.0) # Scale down cost to levels near the ones found in output differential (e.g. 10 */ 10)
    earnedReward = 10*(1/(2**((totalCost**2)/50 + (deltaFreq**2)/2)))
    return earnedReward
