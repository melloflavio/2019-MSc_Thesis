import collections
from typing import  NamedTuple

class CostProfile(NamedTuple):
  alpha: float
  beta: float
  gamma: float

COST_PRESETS = collections.namedtuple('cost_presets', ['COAL', 'OIL'])(
    COAL=CostProfile(alpha=510.0, beta=7.2, gamma=0.00142),
    OIL=CostProfile(alpha=310.0, beta=7.85, gamma=0.00194),
)
