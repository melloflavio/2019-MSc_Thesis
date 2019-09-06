from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from .plot_constants import COLOR_PALETTE, FIG_SIZE, FONT_SIZES

def plotRewardComponents(rewardDictList: List[Dict[str, float]], figureNum=0):
  # details list is a collection of dicts. Needs to be consolidated into a dict of lists
  df = pd.DataFrame(rewardDictList)
  rewardComponents = df.to_dict(orient='list')

  # New plot figure
  plt.figure(figureNum, figsize=FIG_SIZE)

  # Plot each line
  for index, (componentName, valueList) in enumerate(rewardComponents.items()):
    steps = range(len(valueList))
    plotColor = COLOR_PALETTE[index%len(COLOR_PALETTE)]
    plt.plot(steps, valueList, color=plotColor, label=componentName)

  plt.legend()
  plt.xlabel('Steps', fontsize=FONT_SIZES['AXIS_LABEL'])
  plt.ylabel('Reward', fontsize=FONT_SIZES['AXIS_LABEL'])

  plt.title('Reward Components x Time (Steps)', fontsize=FONT_SIZES['TITLE'])

  plt.show()

def plotExperimentRewardProgression(allRewards):

  for idx, rewardDictList in enumerate(allRewards):
    plotRewardComponents(rewardDictList=rewardDictList, figureNum=idx)
