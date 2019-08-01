import matplotlib.pyplot as plt

from dto import SystemHistory

from .color_palette import COLOR_PALETTE

def plot_total_costs(figureNum=0):

  # Get series to be plotted
  stepsSeries = SystemHistory().steps
  actualCostsSeries = SystemHistory().totalCosts['Actual']
  minCostsSeries = SystemHistory().totalCosts['Minimum']
  # How much actual costs are over minimum in %
  deltaCostSeries = [((actual - minimum)/minimum)*100 for (actual, minimum) in zip(actualCostsSeries, minCostsSeries)]

  plt.figure(figureNum)

  # Declare colors to be used
  colorCostsActual = COLOR_PALETTE[0]
  colorCostsMinimum = COLOR_PALETTE[1]
  colorCostsDifference = COLOR_PALETTE[2]

  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()
  # Plot actual and minimum costs in main, left axis
  ax1.plot(stepsSeries, actualCostsSeries, color=colorCostsActual)
  ax1.plot(stepsSeries, minCostsSeries, color=colorCostsMinimum)

  # Plot cost difference in right-side axis
  ax2.plot(stepsSeries, deltaCostSeries, color=colorCostsDifference, linestyle='--')

  ax1.legend(['Actual Cost', 'Minimum Cost'])
  ax1.set_xlabel('Steps', fontsize=12)
  ax1.set_ylabel('Cost ($)', fontsize=12)
  ax2.set_ylabel('Cost Differential (%)', color=colorCostsDifference, fontsize=12)

  plt.title('Costs ($) x Time (Steps)', fontsize=14)

  plt.show()
