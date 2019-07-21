class area:

  def __init__(self, generators, loads, inertia, dampening, timeConstant, droop, nominalFrequency, initialFrequency):
    self.inertia = inertia            # inertia => M
    self.dampening = dampening        # dampening => D
    self.timeConstant = timeConstant  # timeConstant => Tg
    self.droop = droop                # droop => Rd
    self.nominalFrequency = nominalFrequency # nominalFrequency = f or f nom

    self.frequency = initialFrequency
