import tensorflow as tf

from electricity import ElectricalSystem

from .learning_agent import Agent

class ModelTrainer():
  @staticmethod
  def trainAgents(electricalSystem: ElectricalSystem):
    agents = [Agent(_id) for _id in electricalSystem.getGeneratorIds()]
