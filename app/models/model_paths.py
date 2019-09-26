import os

MODEL_FILENAME = 'model'
PARAMS_FILENAME = 'learning_params.json'


def getPathForModel(modelName):
  basePath = os.path.dirname(__file__)
  finalPath = os.path.join(basePath, modelName, MODEL_FILENAME)
  return finalPath

def getPathForParams(modelName):
  basePath = os.path.dirname(__file__)
  finalPath = os.path.join(basePath, modelName, PARAMS_FILENAME)
  return finalPath
