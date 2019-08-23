import os

def getPathForModel(modelName):
  basePath = os.path.dirname(__file__)
  finalPath = os.path.join(basePath, modelName, 'model')
  return finalPath
