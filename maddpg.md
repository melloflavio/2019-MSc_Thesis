```python
# Sample the experience batch (mini batch)
experienceBatch = EpisodeBuffer.sampleExperienceBatch()

# Get Target Actors' Actions
allTargetActions = TargetActors.calculateActionsForBatch(experienceBatch)

# Get Target Critics' Q Estimations
allCriticTargetQs = TargetCritics.calculateQvalsForBatch(experienceBatch, allTargetActions)

# Update the critic networks with the new Q's
Critics.updateModelsForBatch(experienceBatch, allCriticTargetQs)

# Calculate actions for all actors
allActions = Actors.calculateActionsForBatch(experienceBatch)

# Calculate the critic's gradients from the estimated actions
allGradients = Critics.calculateGradientsForBatch(experienceBatch, allActions)

# Update the actor models with the gradients calculated by the critics
Actors.updateModelsForBatch(experienceBatch, allGradients)

# Update target actor and critic models for all agents
TargetActors.updateModels(Actors, tau)
TargetCritics.updateModels(Critics, tau)
```
