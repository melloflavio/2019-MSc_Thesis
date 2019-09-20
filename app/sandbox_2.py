# from electricity import ElectricalSystem, Load, Generator
from dto import (
    COST_PRESETS,
    ElectricalSystemSpecs,
    EpsilonSpecs,
    GeneratorSpecs,
    LoadSpecs,
    NodeStatePower,
    SystemHistory,
    )

trackCostSpecs = ElectricalSystemSpecs(
        loads=[LoadSpecs(id_="L1", basePower=4.0, noiseLevel=0.05)],
        generators=[
            # GeneratorSpecs(id_="G1", basePower=1.33, costProfile=COST_PRESETS.COAL, minPower=0.5, maxPower=5.0, noiseLevel=0.05),
            # GeneratorSpecs(id_="G2", basePower=1.33, costProfile=COST_PRESETS.OIL, minPower=0.5, maxPower=5.0, noiseLevel=0.05),
            # GeneratorSpecs(id_="G3", basePower=1.33, costProfile=COST_PRESETS.GAS, minPower=0.5, maxPower=2.0, noiseLevel=0.05),
            GeneratorSpecs(id_="G1", basePower=2.5, costProfile=COST_PRESETS.COAL_2, minPower=0.5, maxPower=2.5, noiseLevel=0.05),
            GeneratorSpecs(id_="G2", basePower=1, costProfile=COST_PRESETS.OIL_2, minPower=0.5, maxPower=2.5, noiseLevel=0.05),
            GeneratorSpecs(id_="G3", basePower=1, costProfile=COST_PRESETS.OIL_ALTERNATE_2, minPower=0.5, maxPower=2.5, noiseLevel=0.05),
        ],
        shouldTrackOptimalCost=True,
    )

from electricity import ElectricalSystemFactory
costElectricalSystem = ElectricalSystemFactory.create(trackCostSpecs)
