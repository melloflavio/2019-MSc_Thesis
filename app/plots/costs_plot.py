import matplotlib.pyplot as plt

from dto import SystemHistory

from .plot_constants import COLOR_PALETTE, FONT_SIZES

def plotTotalCosts(history: SystemHistory, figureNum=0):

  # Get series to be plotted
  stepsSeries = history.steps
  actualCostsSeries = history.totalCosts['Actual']
  minCostsSeries = history.totalCosts['Minimum']
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

  ax1.set_xlabel('Steps', fontsize=FONT_SIZES['AXIS_LABEL'])
  ax1.set_ylabel('Cost ($)', fontsize=FONT_SIZES['AXIS_LABEL'])
  ax2.set_ylabel('Relative Cost to Optimal (%)', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.title('Costs ($) x Time (Steps)', fontsize=FONT_SIZES['TITLE'])

  plt.show()

def plotIndividualCostsAbsolute(history: SystemHistory, figureNum=0):

  # Get series to be plotted
  stepsSeries = history.steps
  actualCosts = history.actualCosts
  optimalCosts = history.costOptimalCosts

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
  plt.xlabel('Steps', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Cost ($)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('Per Generator Costs ($) x Time (Steps)', fontsize=FONT_SIZES['TITLE'])

  plt.show()

def plotIndividualCostsRelative(history: SystemHistory, figureNum=0):

  # Get series to be plotted
  stepsSeries = history.steps
  actualCosts = history.actualCosts
  optimalCosts = history.costOptimalCosts

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
  plt.xlabel('Steps', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Cost Differential (%)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('Per Generator Costs Differential (%) x Time (Steps)', fontsize=FONT_SIZES['TITLE'])

  plt.show()
