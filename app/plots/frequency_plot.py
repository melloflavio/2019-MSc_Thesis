import matplotlib.pyplot as plt
import numpy as np

from dto import ElectricalConstants, SystemHistory

from .plot_constants import COLOR_PALETTE, FIG_SIZE, FONT_SIZES, LINE_WIDTH

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

  plt.xlabel('Time (s)', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('System Frequency (Hz)', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('System Frequency (Hz) x Time (s)', fontsize=FONT_SIZES['TITLE'])

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
    colotAccepetdRegion = 'green'

    # Plot data
    plt.plot(stepsSeries, nominalSeries, color=colorNominalFreq, label='Nominal', linewidth=LINE_WIDTH)
    plt.plot(stepsSeries, frequencySeries, color=colorObservedFreq, label='Observed', linewidth=LINE_WIDTH)

    plt.xlabel('Time (s)', fontsize=FONT_SIZES['AXIS_LABEL'])
    plt.ylabel('System Frequency (Hz)', fontsize=FONT_SIZES['AXIS_LABEL'])

    DEVIATION = 0.24
    plt.ylim(bottom=nominalFrequency-DEVIATION, top=nominalFrequency+DEVIATION) # Set zoomed in y limits
    plt.yticks(np.arange(nominalFrequency-DEVIATION, nominalFrequency+DEVIATION, (2*DEVIATION)/8))

    ACCEPTED_RANGE = 0.02
    plt.fill_between(
      stepsSeries,
      y1=nominalFrequency-ACCEPTED_RANGE,
      y2=nominalFrequency+ACCEPTED_RANGE,
      color=colotAccepetdRegion,
      alpha=0.25)
    plt.fill(np.NaN, np.NaN, colotAccepetdRegion, alpha=0.25, label='AGC Range')

    plt.legend(loc='upper right')

    plt.title('System Frequency (Hz) x Time (s)', fontsize=FONT_SIZES['TITLE'])

    plt.show()
