from dto import SystemHistory

from .frequency_plot import plotFrequency, plotFrequencyZoom
from .observed_power_plot import plotObservedPower, plotIndividualPowerVsOptimal
from .costs_plot import plotTotalCosts, plotIndividualCostsAbsolute, plotIndividualCostsRelative

def plotAll(history: SystemHistory):
  ## Primary+Secondary Control
  # Plot System Frequency vs Setpoint
  plotFrequency(history, 0)
  plotFrequencyZoom(history, 1)

  # Plot Observed system power output (+ individual generators)
  plotObservedPower(history, 2, shouldPlotAllLoads=False)

  plotIndividualPowerVsOptimal(history, 3)

  ## Tertirary Control
  # Plot Total Costs: Observed v Minimum
  plotTotalCosts(history, 4)

  # Plot Individual Costs: Observed v Minimum
  plotIndividualCostsAbsolute(history, 5)

  # Plot Total Costs: % difference from Observed to Minimum
  plotIndividualCostsRelative(history, 6)
