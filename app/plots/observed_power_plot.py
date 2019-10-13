import matplotlib.pyplot as plt

from dto import SystemHistory

from .plot_constants import COLOR_PALETTE, FIG_SIZE, FONT_SIZES

def plotObservedPower(history: SystemHistory, figureNum=0, shouldPlotAllLoads=False):

  # Get series to be plotted
  stepsSeries = history.steps
  loads = history.loads
  totalLoadSeries = history.totalLoad
  generators = history.generators
  totalPowerSeries = history.totalPower

  plt.figure(figureNum, figsize=FIG_SIZE)

  # Declare colors to be used
  colorTotalLoad = COLOR_PALETTE[0]
  colorTotalPower = COLOR_PALETTE[1]
  colorsIndividualNodes = COLOR_PALETTE[2:]

  # Plot total power/load data
  plt.plot(stepsSeries, totalLoadSeries, color=colorTotalLoad, label='Total Load')
  plt.plot(stepsSeries, totalPowerSeries, color=colorTotalPower, label='Observed Power')

  # Multiple scenarios involve a single load, so we make it optional to plot that single load
  if(shouldPlotAllLoads):
    for idx, loadId in enumerate(loads):
      # Since num loads is variable, colors may wrap around the palette
      loadColor = colorsIndividualNodes[idx % len(colorsIndividualNodes)]
      loadSeries = loads[loadId]
      plt.plot(stepsSeries, loadSeries, color=loadColor, linestyle='--', label=f'{loadId}')

  for idx, generatorId in enumerate(generators):
    # Since num generators is variable, colors may wrap around the palette
    generatorColor = colorsIndividualNodes[idx % len(colorsIndividualNodes)]
    generatorSeries = generators[generatorId]
    plt.plot(stepsSeries, generatorSeries, color=generatorColor, label=f'{generatorId}')

  # totalSecondary = [sum (perGenOutput) for perGenOutput in zip(*generators.values())]
  # plt.plot(stepsSeries, totalSecondary, color=colorTotalPower, label='Total Secondary Ouput', linestyle='--')

  plt.legend()
  plt.xlabel('Time (s)', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Power (pu)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('System Power (pu) x Time (s)', fontsize=FONT_SIZES['TITLE'])

  plt.show()

def plotObservedPowerGenerators(history: SystemHistory, figureNum=0):

  # Get series to be plotted
  stepsSeries = history.steps
  generators = history.generators

  plt.figure(figureNum, figsize=FIG_SIZE)

  # Declare colors to be used
  colorsIndividualNodes = COLOR_PALETTE

  for idx, generatorId in enumerate(generators):
    # Since num generators is variable, colors may wrap around the palette
    generatorColor = colorsIndividualNodes[idx % len(colorsIndividualNodes)]
    generatorSeries = generators[generatorId]
    plt.plot(stepsSeries, generatorSeries, color=generatorColor, label=f'{generatorId}')

  plt.legend()
  plt.xlabel('Time (s)', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Power (pu)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('Per Generator - Power (pu) x Time (s)', fontsize=FONT_SIZES['TITLE'])

  plt.show()

def plotObservedPowerZoomed(history: SystemHistory, figureNum=0):

  # Get series to be plotted
  stepsSeries = history.steps
  loads = history.loads
  totalLoadSeries = history.totalLoad
  generators = history.generators
  totalPowerSeries = history.totalPower

  plt.figure(figureNum, figsize=FIG_SIZE)

  # Declare colors to be used
  colorTotalLoad = COLOR_PALETTE[0]
  colorTotalPower = COLOR_PALETTE[1]
  colorsIndividualNodes = COLOR_PALETTE[2:]

  # Plot total power/load data
  plt.plot(stepsSeries, totalLoadSeries, color=colorTotalLoad, label='Total Load')
  plt.plot(stepsSeries, totalPowerSeries, color=colorTotalPower, label='Observed Power')

  totalSecondary = [sum (perGenOutput) for perGenOutput in zip(*generators.values())]
  plt.plot(stepsSeries, totalSecondary, color=colorTotalPower, label='Total Secondary Ouput', linestyle='--')

  totalLoad = totalLoadSeries[0]
  DEVIATION = 1
  plt.ylim(bottom=totalLoad-DEVIATION, top=totalLoad+DEVIATION) # Set zoomed in y limits

  plt.legend()
  plt.xlabel('Time (s)', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Power (pu)', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.title('System Power (pu) x Time (s) - Zoom', fontsize=FONT_SIZES['TITLE'])

  plt.show()

def plotIndividualPowerVsOptimal(history: SystemHistory, figureNum=0):

  # Get series to be plotted
  stepsSeries = history.steps
  actualPower = history.generators
  optimalPower = history.costOptimalPowers

  plt.figure(figureNum, figsize=(7, 4))

  for idx, generatorId in enumerate(actualPower):
    # Since num generators is variable, colors may wrap around the palette
    generatorColor = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    actualPowerSeries = actualPower[generatorId]
    optimalPowerSeries = optimalPower[generatorId]
    plt.plot(stepsSeries, actualPowerSeries, color=generatorColor, linestyle='-', label=f'{generatorId} Actual')
    plt.plot(stepsSeries, optimalPowerSeries, color=generatorColor, linestyle='--', label=f'{generatorId} Optimal')

  plt.legend(loc='right')
  plt.xlabel('Time (s)', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Power (pu)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('Actual vs Optimal Per Generator Output (pu) x Time (s)', fontsize=FONT_SIZES['TITLE'])

  plt.show()

def plotIndividualPowerVsInitialOptimal(history: SystemHistory, figureNum=0):

  # Get series to be plotted
  stepsSeries = history.steps
  actualPower = history.generators
  optimalPower = history.costOptimalPowers

  plt.figure(figureNum, figsize=FIG_SIZE)

  for idx, generatorId in enumerate(actualPower):
    # Since num generators is variable, colors may wrap around the palette
    generatorColor = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    actualPowerSeries = actualPower[generatorId]
    optimalPowerInitial = optimalPower[generatorId][0]
    optimalPowerSeries = [optimalPowerInitial]*len(stepsSeries)
    plt.plot(stepsSeries, actualPowerSeries, color=generatorColor, linestyle='-', label=f'{generatorId} Actual')
    plt.plot(stepsSeries, optimalPowerSeries, color=generatorColor, linestyle='--', label=f'{generatorId} Optimal')

  plt.legend()
  plt.xlabel('Time (s)', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Power (pu)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('Per Generator Output (pu) Actual vs Target x Time (s)', fontsize=FONT_SIZES['TITLE'])

  plt.show()

def plotPowerDifferentialFromInitialOptimal(history: SystemHistory, figureNum=0):
  """Analogous to ElectricalSystem.getOptimalDifferentialFromInitialState """


  # Get series to be plotted
  stepsSeries = history.steps
  actualPower = history.generators
  optimalPower = history.costOptimalPowers

  plt.figure(figureNum, figsize=FIG_SIZE)

  totalDiffColor = COLOR_PALETTE[0]

  allOutputDifferentials = []
  for idx, generatorId in enumerate(actualPower):
    # Since num generators is variable, colors may wrap around the palette
    actualPowerSeries = actualPower[generatorId]
    optimalPowerInitial = optimalPower[generatorId][0]
    powerDifferentialSeries = [abs((actualOutput/optimalPowerInitial - 1)*100) for actualOutput in actualPowerSeries]

    allOutputDifferentials.append(powerDifferentialSeries)

  allDifferentialSeries = [sum (stepOutputDifferentials) for stepOutputDifferentials in zip(*allOutputDifferentials)]


  plt.plot(stepsSeries, allDifferentialSeries, color=totalDiffColor, linestyle='-', label=f'Output Differential')

  plt.legend()
  plt.xlabel('Time (s)', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Power Differential From Optimal (%)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('Aggregated Power Differential From Optimal (%) x Time (s)', fontsize=FONT_SIZES['TITLE'])

  plt.show()
