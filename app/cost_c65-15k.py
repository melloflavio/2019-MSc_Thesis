from learning  import LearningParams, ModelTrainer, ModelTester
from learning.cost import ModelAdapterCost as ModelAdapter
from dto import (
    COST_PRESETS,
    ElectricalSystemSpecs,
    EpsilonSpecs,
    GeneratorSpecs,
    LoadSpecs,
    NodeStatePower,
    SystemHistory,
    )


from dto import EpsilonSpecs, ElectricalConstants
import math

LearningParams(
    gamma=0.9,   # Gamma (Discount)
    tau=0.001, # Tau
    epsilonSpecs = EpsilonSpecs( # Epsilon explore/exploit control
        thresholdProgress = 0.6, # % of steps where decay change should happen
        thresholdValue = 0.5, # Value at which decay change would happen
        finalValue = 0.0001, # Value at the end of the experiment
    ),
    numEpisodes=15000, # Number of learning episodes to run
    maxSteps=100,   # Number of steps per learning episode
    bufferSize=500, # Experience Buffer Size
    traceLength=15,     # Number of steps each sampled episode should contain
    batchSize=4,     # Number of episodes sampled from experience buffer
    updateInterval=4, # Run update cycle every N steps
    electricalSystemSpecs = ElectricalSystemSpecs(
        loads=[LoadSpecs(id_="L1", basePower=3.0, noiseLevel=0.05)],
        generators=[
            GeneratorSpecs(id_="G1", basePower=2, costProfile=COST_PRESETS.COAL_OLD, minPower=0.5, maxPower=3.0, noiseLevel=0.1),
            GeneratorSpecs(id_="G2", basePower=2, costProfile=COST_PRESETS.OIL_OLD, minPower=0.5, maxPower=3.0, noiseLevel=0.1),
#             GeneratorSpecs(id_="G3", basePower=1.0, costProfile=COST_PRESETS.OIL_ALTERNATE_2, minPower=0.5, maxPower=5.0, noiseLevel=0.05),
        ],
        shouldTrackOptimalCost=True,
    ),
    modelName='2Gen-C65-15k'
)

def rewardFn(totalCost, outputDifferential):
    scaledCost = totalCost/(10000*100.0)
    costComponent = 2**(-1*(scaledCost**2)*200)
    outputComponent = 2**(-1*(outputDifferential**2)*500)
    earnedReward = costComponent*outputComponent

    # print(f'Scaled_Cost:{scaledCost} - OutputDiff:{outputDifferential} - costComponent:{costComponent} - outputComponent:{outputComponent} - reward:{earnedReward}')

    return earnedReward, {'cost': costComponent, 'output': outputComponent, 'total':earnedReward}

modelAdapter = ModelAdapter(rewardFn=rewardFn)
agents = ModelTrainer(modelAdapter).trainAgents()
