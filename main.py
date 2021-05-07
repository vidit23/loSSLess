import barlow
import finetune
import postTraining
import torch

backboneState = barlow.getTrainedBarlowModel(1000)
print("Barlow has finished training the backbone")

finetunedModel = finetune.finetuneBackbone(backboneState, 100)
print("Finetuning on the training data is completed")

activeTrainModel = postTraining.unlabeledActiveLearning(finetunedModel)
print("Predicting on unlabeled data and then training using that is completed")

torch.save(activeTrainModel, "./model.pth")
print("Model saved")