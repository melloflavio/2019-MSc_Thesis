import collections
from typing import  NamedTuple

class CostProfile(NamedTuple):
  alpha: float
  beta: float
  gamma: float

COST_PRESETS = collections.namedtuple('cost_presets', [
  'COAL_2',
  'OIL_2',
  'OIL_ALTERNATE_2',
  'COAL', 'OIL', 'GAS', 'COAL_OLD', 'OIL_OLD'])(
    COAL_2=CostProfile(alpha=510.0, beta=7.7, gamma=0.00142),
    OIL_2=CostProfile(alpha=310.0, beta=7.85, gamma=0.00194),
    OIL_ALTERNATE_2=CostProfile(alpha=78.0, beta=7.55, gamma=0.00482),
    COAL=CostProfile(alpha=300.0, beta=5, gamma=0.00100),
    OIL=CostProfile(alpha=500.0, beta=8, gamma=0.00194),
    GAS=CostProfile(alpha=700.0, beta=10, gamma=0.00482),
    COAL_OLD=CostProfile(alpha=0.0, beta=0, gamma=1),
    OIL_OLD=CostProfile(alpha=0.0, beta=0, gamma=2),
)
