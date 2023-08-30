from enum import Enum

import torch
import torchvision.models as tm
from torch import Tensor
from torch import nn
from torchvision import transforms

from ehrapylat.latent_extractors.predictors.models.BaseExtractor import (
    BaseExtractor as BE,
)
from ehrapylat.constants import CHECKPOINT_DIR
from ehrapylat.latent_extractors.predictors.models.utils import ResNetInit, ResNetModels
from timm.data import auto_augment


class Identity(nn.Module):
    """An identity class to replace arbitrary layers in pretrained models.
    Example::
        from pl_bolts.utils import Identity
        model = resnet18()
        model.fc = Identity()
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class fc_block(nn.Module):
    """An identity class to replace arbitrary layers in pretrained models.
    Example::
        from pl_bolts.utils import Identity
        model = resnet18()
        model.fc = Identity()
    """

    def __init__(self, in_features=512, num_classes=5) -> None:
        super().__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, in_features // 8),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(in_features // 8, num_classes)

    def forward(self, x: torch.Tensor, return_class: bool = False) -> torch.Tensor:
        x_lat = self.fc_seq(x)
        if return_class:
            out = self.out_layer(x_lat)
            return out
        else:
            return x_lat


class ResNetAdapted(nn.Module):
    def __init__(self, model, fc, num_classes=5):
        super().__init__()
        self.model = model
        self.model.fc = Identity()
        self.new_fc = fc
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, return_class: bool = False) -> torch.Tensor:
        x = self.model(x)
        out = self.new_fc(x, return_class=return_class)
        return out

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResNetFromKagglePretrained(nn.Module):
    def __init__(self, model, num_classes=5):
        super().__init__()
        self.model = model
        self.last_fc = self.model.fc[-1]
        self.model.fc = nn.Sequential(model.fc[0:6])
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, return_class: bool = False) -> torch.Tensor:
        x = self.model(x)
        if return_class:
            out = self.last_fc(x)
            return out
        else:
            return x


class ResNetFCWrapper(BE):
    """
    Wrapper around Resnet model from torchvision
    """

    def __init__(
        self,
        model,
        model_name: str,
        initialization: ResNetInit,
        module: str,
        model_id: str,
        num_classes=5,
        opts=None,
    ):
        self.initialization = initialization
        self.num_classes = num_classes
        self.model_name = model_name
        # model = self.get_model(model_name)
        self._set_transform()
        super().__init__(
            model_name=model_name,
            model=model,
            module=module,
            model_id=model_id,
            opts=opts,
        )

    @classmethod
    def init_for_training(
        cls,
        model_name: str,
        initialization: ResNetInit,
        module: str,
        model_id: str,
        num_classes=5,
        opts=None,
    ):
        model = cls.get_model(
            cls,
            model_name=model_name,
            initialization=initialization,
            num_classes=num_classes,
        )
        return cls(
            model=model,
            model_name=model_name,
            initialization=initialization,
            module=module,
            model_id=model_id,
            num_classes=num_classes,
            opts=opts,
        )

    def get_model(self, model_name, initialization, num_classes=5):
        if initialization == ResNetInit.ImageNet:
            model = getattr(tm, model_name)(pretrained=True)

            fc = fc_block(in_features=model.fc.in_features, num_classes=num_classes)
            model = ResNetAdapted(model, fc, num_classes=num_classes)

        elif initialization == ResNetInit.HistDINO:
            model = getattr(tm, model_name)(pretrained=False)
            model.fc = Identity()
            model_path = ResNetFCWrapper.get_pretrained_model_path(
                initialization=initialization
            )
            model = ResNetFCWrapper._load_pretrained(
                model=model, weight_path=model_path
            )

            fc = fc_block(in_features=model.fc.in_features, num_classes=num_classes)
            model = ResNetAdapted(model, fc, num_classes=num_classes)

        elif initialization == ResNetInit.Raw:
            model = getattr(tm, model_name)(pretrained=False)

            fc = fc_block(in_features=model.fc.in_features, num_classes=num_classes)
            model = ResNetAdapted(model, fc, num_classes=num_classes)

        elif initialization == ResNetInit.ActivationBased:
            model = getattr(tm, model_name)(pretrained=False)
            fc = fc_block(in_features=model.fc.in_features, num_classes=num_classes)
            model = ResNetAdapted(model, fc)
            model._init_params()
            return model
        elif initialization == ResNetInit.Kaggle:
            model_path = ResNetFCWrapper.get_pretrained_model_path(
                initialization=initialization
            )
            model = ResNetFCWrapper._load_pretrained(model=None, weight_path=model_path)
            model = ResNetFromKagglePretrained(model=model)

        return model

    @staticmethod
    def get_pretrained_model_path(initialization=ResNetInit.HistDINO):
        if initialization == ResNetInit.HistDINO:
            model_path = CHECKPOINT_DIR / "DINO" / "resnet50_tcga_brca_simclr.pt"
        elif initialization == ResNetInit.Kaggle:
            model_path = CHECKPOINT_DIR / "kaggle" / "kaggle_weights.pth"
        assert model_path.exists()
        return model_path

    @staticmethod
    def _load_pretrained(model=None, weight_path=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is not None:
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            model = torch.load(weight_path, map_location=device)
        return model

    def _set_transform(self):
        if self.initialization == ResNetInit.HistDINO:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _set_augmentations(self):
        augmentations = auto_augment.rand_augment_transform(
            config_str="rand-m9-mstd0.5", hparams={}
        )
        if self.initialization == ResNetInit.HistDINO:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            self.train_transforms = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToPILImage(),
                    augmentations,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.train_transforms = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToPILImage(),
                    augmentations,
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def forward(self, imgs, return_class=False):
        # imgs = self.transform(imgs)
        outputs = self.model(imgs, return_class=return_class)
        return outputs
