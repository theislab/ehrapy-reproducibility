from torch.optim import lr_scheduler
from enum import Enum
import pytorch_lightning as pl
import os
from torch.optim.lr_scheduler import CyclicLR


class ResNetInit(Enum):
    ImageNet = "ImageNet"
    HistDINO = "HistDINO"
    Raw = "Raw"
    ActivationBased = "ActivationBased"
    Kaggle = "Kaggle"


class ResNetModels(Enum):
    resnet18 = "resnet18"
    resnet34 = "resnet34"
    resnet50 = "resnet50"
