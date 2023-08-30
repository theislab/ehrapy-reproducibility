from enum import Enum

import torch
from torchvision import transforms

from ehrapylat.latent_extractors.predictors.models.BaseExtractor import (
    BaseExtractor as BE,
)
from ehrapylat.latent_extractors.predictors.models.SpatialSSL.modules.vit_dino import (
    vit_small,
)
from ehrapylat.constants import CHECKPOINT_DIR


class VitDINOModels(Enum):
    vit_small = "vit_small"


class VitDINOWrapper(BE):
    """
    Wrapper around Resnet model from torchvision
    """

    def __init__(self, model_name: str, module: str, model_id: str, opts=None):
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
        model = vit_small(patch_size=16)
        model_path = self.get_pretrained_model_path(model_name)
        model = self._load_pretrained(model, model_path)
        return model

    def get_pretrained_model_path(self, model_name):
        model_path = CHECKPOINT_DIR / "DINO" / "vits_tcga_brca_dino.pt"
        assert model_path.exists()
        return model_path

    def _load_pretrained(self, model, weight_path):
        state_dict = torch.load(weight_path, map_location="cpu")["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        return model

    def _set_transform(self):
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.Normalize(mean=mean, std=std)]
        )

    def forward(self, imgs):
        # imgs = self.transform(imgs)
        outputs = self.model(imgs)
        return outputs
