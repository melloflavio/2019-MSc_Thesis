import matplotlib.pyplot as plt

from dto import ElectricalConstants, SystemHistory

from .plot_constants import COLOR_PALETTE, FIG_SIZE, FONT_SIZES

def plotFrequency(history: SystemHistory, figureNum=0):

  # Get series to be plotted
  frequencySeries = history.frequency
  stepsSeries = history.steps
  nominalFrequency = ElectricalConstants().nominalFrequency
  nominalSeries = [nominalFrequency] * len(stepsSeries)

  plt.figure(figureNum, figsize=FIG_SIZE)

  # Declare colors to be used
  colorNominalFreq = COLOR_PALETTE[0]
  colorObservedFreq = COLOR_PALETTE[1]

  # Plot data
  plt.plot(stepsSeries, nominalSeries, color=colorNominalFreq, label='Nominal')
  plt.plot(stepsSeries, frequencySeries, color=colorObservedFreq, label='Observed')
  plt.legend()

  plt.xlabel('Steps', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('System Frequency (Hz)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('System Frequency (Hz) x Time (Steps)', fontsize=FONT_SIZES['TITLE'])

  plt.show()

def plotFrequencyZoom(history: SystemHistory, figureNum=0):

    # Get series to be plotted
    frequencySeries = history.frequency
    stepsSeries = history.steps
    nominalFrequency = ElectricalConstants().nominalFrequency
    nominalSeries = [nominalFrequency] * len(stepsSeries)

    plt.figure(figureNum, figsize=FIG_SIZE)

    # Declare colors to be used
    colorNominalFreq = COLOR_PALETTE[0]
    colorObservedFreq = COLOR_PALETTE[1]

    # Plot data
    plt.plot(stepsSeries, nominalSeries, color=colorNominalFreq, label='Nominal')
    plt.plot(stepsSeries, frequencySeries, color=colorObservedFreq, label='Observed')
    plt.legend()

    plt.xlabel('Steps', fontsize=FONT_SIZES['AXIS_LABEL'])
    plt.ylabel('System Frequency (Hz)', fontsize=FONT_SIZES['AXIS_LABEL'])

    DEVIATION = 0.5
    plt.ylim(bottom=nominalFrequency-DEVIATION, top=nominalFrequency+DEVIATION) # Set zoomed in y limits

    plt.title('System Frequency (Hz) x Time (Steps) - Zoom', fontsize=FONT_SIZES['TITLE'])

    plt.show()
