from dto import NodeState

class Load:
  def __init__(self, loadId, initialLoad):
    self.loadId = loadId
    self.load = initialLoad # Load => P_Li

  def getId(self):
    return self.loadId

  def setLoad(self, newLoad):
    self.load = newLoad

  def getLoad(self):
    return self.load

  def toNodeState(self) -> NodeState:
    return NodeState(id_=self.loadId, power=self.getLoad())
