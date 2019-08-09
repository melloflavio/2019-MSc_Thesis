from dto import SystemHistory

from .frequency_plot import plotFrequency
from .observed_power_plot import plotObservedPower
from .costs_plot import plotTotalCosts, plotIndividualCostsAbsolute, plotIndividualCostsRelative

def plotAll(history: SystemHistory):
  ## Primary+Secondary Control
  # Plot System Frequency vs Setpoint
  plotFrequency(history, 0)

  # Plot Observed system power output (+ individual generators)
  plotObservedPower(history, 1, shouldPlotAllLoads=False)

  ## Tertirary Control
  # Plot Total Costs: Observed v Minimum
  plotTotalCosts(history, 2)

  # Plot Individual Costs: Observed v Minimum
  plotIndividualCostsAbsolute(history, 3)

  # Plot Total Costs: % difference from Observed to Minimum
  plotIndividualCostsRelative(history, 4)
