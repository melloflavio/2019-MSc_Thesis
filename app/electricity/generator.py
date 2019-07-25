class generator:

  def __init__(self, initialOutput):
    self.output = initialOutput # output => P_Gi (or Z in secondary control)

  def setOutput(self, newOutput):
    self.output = newOutput

  def getOutput(self):
    return self.output
