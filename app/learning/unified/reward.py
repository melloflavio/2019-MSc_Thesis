def rewardFunction(deltaFreq, totalCost):
    totalCost = totalCost/(10000.0) # Scale down cost to levels near the ones found in output differential (e.g. 10 */ 10)
    costComponent = 2**(-1*(totalCost**2)/50)
    freqComponent = 2**(-1*(deltaFreq**2)/2)
    earnedReward = 10*costComponent*freqComponent

    return earnedReward, {'cost': costComponent, 'freq': freqComponent, 'total':earnedReward}
