from dataclasses import dataclass

@dataclass
class EpsilonSpecs:
  thresholdProgress: float  # % of steps where decay change should happen
  thresholdValue: float      # Value at which decay change would happen
  finalValue: float     # Value at the end of the experiment
