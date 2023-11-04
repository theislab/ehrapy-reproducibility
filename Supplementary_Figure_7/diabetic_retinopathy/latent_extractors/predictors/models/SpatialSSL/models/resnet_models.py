from enum import Enum

import torch
import torchvision.models as tm
from torch import Tensor
from torchvision import transforms

from ehrapylat.latent_extractors.predictors.models.BaseExtractor import (
    BaseExtractor as BE,
)
from ehrapylat.constants import CHECKPOINT_DIR
from ehrapylat.latent_extractors.predictors.models.utils import ResNetInit, ResNetModels


class Identity(torch.nn.Module):
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


class ResNetWrapper(BE):
    """
    Wrapper around Resnet model from torchvision
    """

    def __init__(
        self,
        model_name: str,
        initialization: ResNetInit,
        module: str,
        model_id: str,
        opts=None,
    ):
        self.initialization = initialization
        model = self.get_model(model_name)
        self._set_transform()
        super().__init__(
            model_name=model_name,
            model=model,
            module=module,
            model_id=model_id,
            opts=opts,
        )

    def get_model(self, model_name):
        if self.initialization == ResNetInit.ImageNet:
            model = getattr(tm, model_name)(pretrained=True)
            model.fc = Identity()
            return model

        elif self.initialization == ResNetInit.HistDINO:
            model = getattr(tm, model_name)(pretrained=False)
            model.fc = Identity()
            model_path = self.get_pretrained_model_path(model_name)
            model = self._load_pretrained(model, model_path)
            return model
        return None

    def get_pretrained_model_path(self, model_name):
        model_path = CHECKPOINT_DIR / "DINO" / "resnet50_tcga_brca_simclr.pt"
        assert model_path.exists()
        return model_path

    def _load_pretrained(self, model, weight_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
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

    def forward(self, imgs):
        # imgs = self.transform(imgs)
        outputs = self.model(imgs)
        return outputs
