import matplotlib.pyplot as plt

from dto import SystemHistory

from .plot_constants import COLOR_PALETTE

def plotObservedPower(figureNum=0, shouldPlotAllLoads=False):

  # Get series to be plotted
  stepsSeries = SystemHistory().steps
  loads = SystemHistory().loads
  totalLoadSeries = SystemHistory().totalLoad
  generators = SystemHistory().generators
  totalPowerSeries = SystemHistory().totalPower

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
  plt.xlabel('Steps', fontsize=12)
  plt.ylabel('Power (pu)', fontsize=12)

  plt.title('System Power (pu) x Time (Steps)', fontsize=14)

  plt.show()
