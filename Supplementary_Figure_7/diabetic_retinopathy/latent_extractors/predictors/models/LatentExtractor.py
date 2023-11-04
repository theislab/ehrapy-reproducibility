from enum import Enum

from ehrapylat.latent_extractors.predictors.models.SpatialSSL.models.dino_models import (
    VitDINOModels,
    VitDINOWrapper,
)
from ehrapylat.latent_extractors.predictors.models.SpatialSSL.models.hf_models import (
    HFModels,
    HFWrapper,
)
from ehrapylat.latent_extractors.predictors.models.SpatialSSL.models.resnet_models import (
    ResNetWrapper,
)
from ehrapylat.latent_extractors.predictors.models.FC_trained_models.resnet_models import (
    ResNetFCWrapper,
)
from ehrapylat.latent_extractors.predictors.models.utils import ResNetInit, ResNetModels


class ExportModules(Enum):
    HF = "HF"
    ResNet = "ResNet"
    DINO = "DINO"
    ResNetFC = "ResNetFC"


class Models(Enum):
    ViTHF = {"module": ExportModules.HF, "model_name": HFModels.ViTModel}
    VanModel = {"module": ExportModules.HF, "model_name": HFModels.VanModel}
    resnet18ImgNet = {
        "module": ExportModules.ResNet,
        "model_name": ResNetModels.resnet18,
        "initialization": ResNetInit.ImageNet,
    }
    resnet34ImgNet = {
        "module": ExportModules.ResNet,
        "model_name": ResNetModels.resnet34,
        "initialization": ResNetInit.ImageNet,
    }
    resnet50ImgNet = {
        "module": ExportModules.ResNet,
        "model_name": ResNetModels.resnet50,
        "initialization": ResNetInit.ImageNet,
    }
    resnet50HistDINO = {
        "module": ExportModules.ResNet,
        "model_name": ResNetModels.resnet50,
        "initialization": ResNetInit.HistDINO,
    }
    ViTHistDINO = {"module": ExportModules.DINO, "model_name": VitDINOModels.vit_small}

    resnet50HistDINOFC = {
        "module": ExportModules.ResNet,
        "model_name": ResNetModels.resnet50,
        "initialization": ResNetInit.HistDINO,
    }

    resnet18ImgNetFC = {
        "module": ExportModules.ResNetFC,
        "model_name": ResNetModels.resnet18,
        "initialization": ResNetInit.ImageNet,
    }

    resnet34ImgNetFC = {
        "module": ExportModules.ResNetFC,
        "model_name": ResNetModels.resnet34,
        "initialization": ResNetInit.ImageNet,
    }

    resnet50ImgNetFC = {
        "module": ExportModules.ResNetFC,
        "model_name": ResNetModels.resnet50,
        "initialization": ResNetInit.ImageNet,
    }

    resnet18RawFC = {
        "module": ExportModules.ResNetFC,
        "model_name": ResNetModels.resnet18,
        "initialization": ResNetInit.Raw,
    }

    resnet34RawFC = {
        "module": ExportModules.ResNetFC,
        "model_name": ResNetModels.resnet34,
        "initialization": ResNetInit.Raw,
    }

    resnet50RawFC = {
        "module": ExportModules.ResNetFC,
        "model_name": ResNetModels.resnet50,
        "initialization": ResNetInit.Raw,
    }

    resnet18ActBasedFC = {
        "module": ExportModules.ResNetFC,
        "model_name": ResNetModels.resnet18,
        "initialization": ResNetInit.ActivationBased,
    }

    resnet34ActBasedFC = {
        "module": ExportModules.ResNetFC,
        "model_name": ResNetModels.resnet34,
        "initialization": ResNetInit.ActivationBased,
    }

    resnet50ActBasedFC = {
        "module": ExportModules.ResNetFC,
        "model_name": ResNetModels.resnet50,
        "initialization": ResNetInit.ActivationBased,
    }

    resnet50KaggleFC = {
        "module": ExportModules.ResNetFC,
        "model_name": ResNetModels.resnet50,
        "initialization": ResNetInit.Kaggle,
    }


def get_latent_extractor(model_name: Models, num_classes=5, opts=None):
    model_config = model_name.value
    if model_config["module"] == ExportModules.HF:
        extractor = HFWrapper(
            model_name=model_config["model_name"].value,
            module=model_config["module"].value,
            model_id=model_name.name,
        )
    elif model_config["module"] == ExportModules.ResNet:
        extractor = ResNetWrapper(
            model_name=model_config["model_name"].value,
            initialization=model_config["initialization"],
            module=model_config["module"].value,
            model_id=model_name.name,
        )
    elif model_config["module"] == ExportModules.DINO:
        extractor = VitDINOWrapper(
            model_name=model_config["model_name"].value,
            module=model_config["module"].value,
            model_id=model_name.name,
        )
    elif model_config["module"] == ExportModules.ResNetFC:
        extractor = ResNetFCWrapper.init_for_training(
            model_name=model_config["model_name"].value,
            initialization=model_config["initialization"],
            module=model_config["module"].value,
            model_id=model_name.name,
            num_classes=5,
            opts=opts,
        )
    else:
        raise NotImplementedError("No such module")
    return extractor
