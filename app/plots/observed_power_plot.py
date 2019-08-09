import matplotlib.pyplot as plt

from dto import SystemHistory

from .plot_constants import COLOR_PALETTE, FONT_SIZES

def plotObservedPower(history: SystemHistory, figureNum=0, shouldPlotAllLoads=False):

  # Get series to be plotted
  stepsSeries = history.steps
  loads = history.loads
  totalLoadSeries = history.totalLoad
  generators = history.generators
  totalPowerSeries = history.totalPower

  plt.figure(figureNum)

  # Declare colors to be used
  colorTotalLoad = COLOR_PALETTE[0]
  colorTotalPower = COLOR_PALETTE[1]
  colorsIndividualNodes = COLOR_PALETTE[2:]

  # Plot total power/load data
  plt.plot(stepsSeries, totalLoadSeries, color=colorTotalLoad)
  plt.plot(stepsSeries, totalPowerSeries, color=colorTotalPower)
  legendFields = ['Total Load', 'Total Power']

  # Multiple scenarios involve a single load, so we make it optional to plot that single load
  if(shouldPlotAllLoads):
    for idx, loadId in enumerate(loads):
      # Since num loads is variable, colors may wrap around the palette
      loadColor = colorsIndividualNodes[idx % len(colorsIndividualNodes)]
      loadLegend = loadId
      loadSeries = loads[loadId]
      plt.plot(stepsSeries, loadSeries, color=loadColor, linestyle='--')
      legendFields.append(loadLegend)

  for idx, generatorId in enumerate(generators):
    # Since num generators is variable, colors may wrap around the palette
    generatorColor = colorsIndividualNodes[idx % len(colorsIndividualNodes)]
    generatorLegend = generatorId
    generatorSeries = generators[generatorId]
    plt.plot(stepsSeries, generatorSeries, color=generatorColor)
    legendFields.append(generatorLegend)


  plt.legend(legendFields)
  plt.xlabel('Steps', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Power (pu)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('System Power (pu) x Time (Steps)', fontsize=FONT_SIZES['TITLE'])

  plt.show()
