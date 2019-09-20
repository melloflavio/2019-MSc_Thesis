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


LearningParams(
    numEpisodes=5000, # Number of learning episodes to run
    maxSteps=100,   # Number of steps per learning episode
    traceLength=8,     # Number of steps each sampled episode should contain
    batchSize=4,     # Number of episodes sampled from experience buffer
    electricalSystemSpecs= ElectricalSystemSpecs(
        loads=[LoadSpecs(id_="L1", basePower=3.15, noiseLevel=0)],
        generators=[
            GeneratorSpecs(id_="G1", basePower=1.0, costProfile=COST_PRESETS.COAL, minPower=0.5, maxPower=3, noiseLevel=0.1),
            GeneratorSpecs(id_="G2", basePower=1.0, costProfile=COST_PRESETS.OIL, minPower=0.5, maxPower=3, noiseLevel=0.1),
            # GeneratorSpecs(id_="G3", basePower=1.0, costProfile=COST_PRESETS.OIL, minPower=0.5, maxPower=3, noiseLevel=0.1),
        ],
        shouldTrackOptimalCost=True,
    ),
    modelName='gcloud',
)

modelAdapter = ModelAdapter()
agents = ModelTrainer(modelAdapter).trainAgents()
