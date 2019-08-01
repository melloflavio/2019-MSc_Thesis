import matplotlib.pyplot as plt

from dto import ElectricalConstants, SystemHistory

from .color_palette import COLOR_PALETTE

def plot_frequency(figure_num=0):

  # Get series to be plotted
  frequencySeries = SystemHistory().frequency
  stepsSeries = SystemHistory().steps
  nominalFrequency = ElectricalConstants().nominalFrequency
  nominalSeries = [nominalFrequency] * len(stepsSeries)

  plt.figure(figure_num)

  # Declare colors to be used
  colorNominalFreq = COLOR_PALETTE[0]
  colorObservedFreq = COLOR_PALETTE[1]

  # Plot data
  plt.plot(stepsSeries, nominalSeries, color=colorNominalFreq)
  plt.plot(stepsSeries, frequencySeries, color=colorObservedFreq)
  plt.legend(['Nominal', 'Observed'])

  plt.xlabel('Steps', fontsize=12)
  plt.ylabel('System Frequency (Hz)', fontsize=12)

  plt.title('System Frequency (Hz) x Time (Steps)', fontsize=14)

  plt.show()
