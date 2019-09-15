import collections
from typing import  NamedTuple

class CostProfile(NamedTuple):
  alpha: float
  beta: float
  gamma: float

COST_PRESETS = collections.namedtuple('cost_presets', ['COAL', 'OIL', 'GAS'])(
    # COAL=CostProfile(alpha=510.0, beta=7.2, gamma=0.00142),
    # OIL=CostProfile(alpha=310.0, beta=7.85, gamma=0.00194),
    # OIL_ALTERNATE(alpha=78.0, beta=7.97, gamma=0.00482),
    COAL=CostProfile(alpha=300.0, beta=5, gamma=0.00100),
    OIL=CostProfile(alpha=500.0, beta=8, gamma=0.00194),
    GAS=CostProfile(alpha=700.0, beta=10, gamma=0.00482),
)
