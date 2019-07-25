class Generator:

  def __init__(self, generatorId, initialOutput):
    self.generatorId = generatorId
    self.output = initialOutput # output => P_Gi (or Z in secondary control)

  def getId(self):
    return self.generatorId

  def setOutput(self, newOutput):
    self.output = newOutput

  def getOutput(self):
    return self.output
