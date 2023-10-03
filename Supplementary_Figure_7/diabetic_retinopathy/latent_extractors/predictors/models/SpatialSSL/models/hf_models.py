from enum import Enum

import torch
import transformers
from transformers import ViTFeatureExtractor

from ehrapylat.latent_extractors.predictors.models.BaseExtractor import (
    BaseExtractor as BE,
)


class HFModels(Enum):
    ViTModel = "ViTModel"
    VanModel = "VanModel"
    ViTMAEModel = "ViTMAEModel"
    FlaxViTModel = "FlaxViTModel"


class HFWrapper(BE):
    def __init__(self, model_name: str, module: str, model_id: str, opts=None):
        self.model_path = self.get_pretrained_model_path(model_name)
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
        model = getattr(transformers, model_name).from_pretrained(self.model_path)
        return model

    def get_pretrained_model_path(self, model_name):
        if model_name == "ViTModel":
            return "google/vit-base-patch16-224-in21k"

        elif model_name == "VanModel":
            return "Visual-Attention-Network/van-base"

        elif model_name == "ViTMAEModel":
            return "facebook/vit-mae-base"

        elif model_name == "FlaxViTModel":
            return "google/vit-base-patch16-224-in21k"

        else:
            raise NotImplementedError("No such model name")

    def _load_pretrained(self, model, model_path):
        pass

    def _set_transform(self):
        self.transform = ViTFeatureExtractor.from_pretrained(self.model_path)

    def forward(self, imgs):
        # imgs = torch.unbind(imgs, dim=0)  # ViTFeatureExtractor requires list of tensors
        # imgs = imgs["pixel_values"]
        # imgs = torch.cat([img.unsqueeze(0) for img in imgs], dim=0)
        outputs = self.model(imgs)
        return outputs.pooler_output
