from dto import SystemHistory

from .frequency_plot import plotFrequency, plotFrequencyZoom
from .observed_power_plot import plotObservedPower
from .costs_plot import plotTotalCosts, plotIndividualCostsAbsolute, plotIndividualCostsRelative

def plotAll(history: SystemHistory):
  ## Primary+Secondary Control
  # Plot System Frequency vs Setpoint
  plotFrequency(history, 0)
  plotFrequencyZoom(history, 1)

  # Plot Observed system power output (+ individual generators)
  plotObservedPower(history, 2, shouldPlotAllLoads=False)

  ## Tertirary Control
  # Plot Total Costs: Observed v Minimum
  plotTotalCosts(history, 3)

  # Plot Individual Costs: Observed v Minimum
  plotIndividualCostsAbsolute(history, 4)

  # Plot Total Costs: % difference from Observed to Minimum
  plotIndividualCostsRelative(history, 5)
