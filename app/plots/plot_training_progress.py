from typing import List, Dict
import matplotlib.pyplot as plt

from dto import SystemHistory

from .frequency_plot import plotFrequency, plotFrequencyZoom
from .observed_power_plot import plotObservedPower
from .costs_plot import plotTotalCosts, plotIndividualCostsAbsolute, plotIndividualCostsRelative
from .rewards_plot import plotRewardComponents

def plotTraningProgress(history: SystemHistory, rewardDictList: List[Dict[str, float]], allRewards: List[float]):

  # Plot individual components earning
  plotRewardComponents(rewardDictList, 0)

  ## Primary+Secondary Control
  # Plot System Frequency vs Setpoint
  plotFrequency(history, 1)
  plotFrequencyZoom(history, 2)

  # Plot Observed system power output (+ individual generators)
  plotObservedPower(history, 3, shouldPlotAllLoads=False)

  ## Tertirary Control
  # Plot Total Costs: Observed v Minimum
  plotTotalCosts(history, 4)

  # Plot Individual Costs: Observed v Minimum
  plotIndividualCostsAbsolute(history, 5)

  # Plot Total Costs: % difference from Observed to Minimum
  plotIndividualCostsRelative(history, 6)

  plt.figure(7)
  plt.scatter(range(len(allRewards)), allRewards)
