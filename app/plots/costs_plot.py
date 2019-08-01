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

  plt.title('Costs ($) x Time (Steps)', fontsize=14)

  plt.show()

def plot_individual_costs_absolute(figureNum=0):

  # Get series to be plotted
  stepsSeries = SystemHistory().steps
  actualCosts = SystemHistory().actualCosts
  optimalCosts = SystemHistory().costOptimalCosts

  plt.figure(figureNum)

  legendFields = []

  for idx, generatorId in enumerate(actualCosts):
    # Since num generators is variable, colors may wrap around the palette
    generatorColor = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    actualCostsSeries = actualCosts[generatorId]
    optimalCostsSeries = optimalCosts[generatorId]
    plt.plot(stepsSeries, actualCostsSeries, color=generatorColor, linestyle='-')
    plt.plot(stepsSeries, optimalCostsSeries, color=generatorColor, linestyle='--')

    legendFields.extend(['{} Actual'.format(generatorId), '{} Optimal'.format(generatorId)])

  plt.legend(legendFields)
  plt.xlabel('Steps', fontsize=12)
  plt.ylabel('Cost ($)', fontsize=12)

  plt.title('Per Generator Costs ($) x Time (Steps)', fontsize=14)

  plt.show()

def plot_individual_costs_relative(figureNum=0):

  # Get series to be plotted
  stepsSeries = SystemHistory().steps
  actualCosts = SystemHistory().actualCosts
  optimalCosts = SystemHistory().costOptimalCosts

  plt.figure(figureNum)

  legendFields = []

  for idx, generatorId in enumerate(actualCosts):
    # Since num generators is variable, colors may wrap around the palette
    generatorColor = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    actualCostsSeries = actualCosts[generatorId]
    optimalCostsSeries = optimalCosts[generatorId]
    deltaCostSeries = [((actual - minimum)/minimum)*100 for (actual, minimum) in zip(actualCostsSeries, optimalCostsSeries)]
    plt.plot(stepsSeries, deltaCostSeries, color=generatorColor, linestyle='-.')

    legendFields.extend(['{} Actual'.format(generatorId), '{} Optimal'.format(generatorId)])

  plt.legend(legendFields)
  plt.xlabel('Steps', fontsize=12)
  plt.ylabel('Cost Differential (%)', fontsize=12)

  plt.title('Per Generator Costs Differential (%) x Time (Steps)', fontsize=14)

  plt.show()
