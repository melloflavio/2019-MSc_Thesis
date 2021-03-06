import matplotlib.pyplot as plt

from dto import SystemHistory

from .plot_constants import COLOR_PALETTE, FIG_SIZE, FONT_SIZES

def plotTotalCosts(history: SystemHistory, figureNum=0):

  # Get series to be plotted
  stepsSeries = history.steps
  actualCostsSeries = history.totalCosts['Actual']
  minCostsSeries = history.totalCosts['Minimum']
  # How much actual costs are over minimum in %
  deltaCostSeries = [((actual - minimum)/minimum)*100 for (actual, minimum) in zip(actualCostsSeries, minCostsSeries)]

  # Declare colors to be used
  colorCostsActual = COLOR_PALETTE[0]
  colorCostsMinimum = COLOR_PALETTE[1]
  colorCostsDifference = COLOR_PALETTE[2]

  fig, ax1 = plt.subplots(num=figureNum, figsize=FIG_SIZE)
  ax2 = ax1.twinx()
  # Plot actual and minimum costs in main, left axis
  ax1.plot(stepsSeries, actualCostsSeries, color=colorCostsActual, label='Actual Cost')
  ax1.plot(stepsSeries, minCostsSeries, color=colorCostsMinimum, label='Minimum Cost')
  ax1.legend(loc='upper left')

  # Plot cost difference in right-side axis
  ax2.plot(stepsSeries, deltaCostSeries, color=colorCostsDifference, linestyle='--', label='Cost Differential')
  ax2.legend(loc='upper right')

  ax1.set_xlabel('Steps', fontsize=FONT_SIZES['AXIS_LABEL'])
  ax1.set_ylabel('Cost ($/h)', fontsize=FONT_SIZES['AXIS_LABEL'])
  ax2.set_ylabel('Relative Cost to Optimal (%)', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.title('Costs ($/h) x Time (Steps)', fontsize=FONT_SIZES['TITLE'])

  plt.show()

def plotIndividualCostsAbsolute(history: SystemHistory, figureNum=0):

  # Get series to be plotted
  stepsSeries = history.steps
  actualCosts = history.actualCosts
  optimalCosts = history.costOptimalCosts

  plt.figure(figureNum, figsize=FIG_SIZE)

  for idx, generatorId in enumerate(actualCosts):
    # Since num generators is variable, colors may wrap around the palette
    generatorColor = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    actualCostsSeries = actualCosts[generatorId]
    optimalCostsSeries = optimalCosts[generatorId]
    plt.plot(stepsSeries, actualCostsSeries, color=generatorColor, linestyle='-', label=f'{generatorId} Actual')
    plt.plot(stepsSeries, optimalCostsSeries, color=generatorColor, linestyle='--', label=f'{generatorId} Optimal')

  plt.legend()
  plt.xlabel('Steps', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Cost ($/h)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('Per Generator Costs ($/h) x Time (Steps)', fontsize=FONT_SIZES['TITLE'])

  plt.show()

def plotIndividualCostsRelative(history: SystemHistory, figureNum=0):

  # Get series to be plotted
  stepsSeries = history.steps
  actualCosts = history.actualCosts
  optimalCosts = history.costOptimalCosts

  plt.figure(figureNum, figsize=FIG_SIZE)

  for idx, generatorId in enumerate(actualCosts):
    # Since num generators is variable, colors may wrap around the palette
    generatorColor = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    actualCostsSeries = actualCosts[generatorId]
    optimalCostsSeries = optimalCosts[generatorId]
    deltaCostSeries = [((actual - minimum)/minimum)*100 for (actual, minimum) in zip(actualCostsSeries, optimalCostsSeries)]
    plt.plot(stepsSeries, deltaCostSeries, color=generatorColor, linestyle='-.', label=f'{generatorId}')

  plt.legend()
  plt.xlabel('Steps', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Cost Differential (%)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('Per Generator Costs Differential (%) x Time (Steps)', fontsize=FONT_SIZES['TITLE'])

  plt.show()


def plotIndividualCostsAbsoluteToInitial(history: SystemHistory, figureNum=0):

  # Get series to be plotted
  stepsSeries = history.steps
  actualCosts = history.actualCosts
  optimalCosts = history.costOptimalCosts

  plt.figure(figureNum, figsize=FIG_SIZE)

  for idx, generatorId in enumerate(actualCosts):
    # Since num generators is variable, colors may wrap around the palette
    generatorColor = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    actualCostsSeries = actualCosts[generatorId]
    optimalCost = optimalCosts[generatorId]
    optimalCostsSeries = optimalCost[0]*len(stepsSeries) # Horizontal line with the optimal cost at initial
    plt.plot(stepsSeries, actualCostsSeries, color=generatorColor, linestyle='-', label=f'{generatorId} Actual')
    plt.plot(stepsSeries, optimalCostsSeries, color=generatorColor, linestyle='--', label=f'{generatorId} Optimal')

  plt.legend()
  plt.xlabel('Steps', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Cost ($/h)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('Per Generator Costs ($/h) x Time (Steps)', fontsize=FONT_SIZES['TITLE'])

  plt.show()


def plotTotalCostDifferential(history: SystemHistory, figureNum=0):

  # Get series to be plotted
  stepsSeries = history.steps
  actualCosts = history.actualCosts
  optimalCosts = history.costOptimalCosts

  plt.figure(figureNum, figsize=FIG_SIZE)

  for idx, generatorId in enumerate(actualCosts):
    # Since num generators is variable, colors may wrap around the palette
    generatorColor = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    actualCostsSeries = actualCosts[generatorId]
    optimalCost = optimalCosts[generatorId]
    optimalCostsSeries = optimalCost[0]*len(stepsSeries) # Horizontal line with the optimal cost at initial
    plt.plot(stepsSeries, actualCostsSeries, color=generatorColor, linestyle='-', label=f'{generatorId} Actual')
    plt.plot(stepsSeries, optimalCostsSeries, color=generatorColor, linestyle='--', label=f'{generatorId} Optimal')

  plt.legend()
  plt.xlabel('Steps', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Cost ($/h)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('Per Generator Costs ($/h) x Time (Steps)', fontsize=FONT_SIZES['TITLE'])

  plt.show()
