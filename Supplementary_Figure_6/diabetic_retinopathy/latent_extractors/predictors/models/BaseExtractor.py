import json
import os
import pickle as pkl

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from tqdm import tqdm

from ehrapylat.constants import RESULTS_DIR
from ehrapylat.latent_extractors.predictors.utils import get_scheduler
import torch.nn.functional as F
from torchmetrics.functional import f1_score, accuracy


class BaseExtractor(pl.LightningModule):
    def __init__(
        self, model_name: str, model: nn.Module, module: str, model_id: str, opts=None
    ):
        """
        Base speleton wrappper class for different models from various sources (HuggingFace, DINO, etc..)
        @param model_name: string corresponding to the model name withing the export model
        @param model: pytorch model
        @param module: specifies the module from which the model is coming from (HF, DINO, etc..)
        @param module: unique model name (listd in the sslbio.models.LatentExtractor.Models)

        """
        super(BaseExtractor, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.model_name = model_name
        self.module = module
        self.model_id = model_id
        if opts is not None:
            self.save_hyperparameters(
                opts.__dict__, ignore=["model_path", "ipython_dir"]
            )
            self.criterion = getattr(nn, self.hparams.nn_loss)()

        # self.res

    def get_model(self, model_name):
        pass

    def _load_pretrained(self):
        pass

    def _set_transform(self):
        pass

    @staticmethod
    def get_pretrained_model_path(model_name):
        pass

    def forward(self, imgs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(0.5, 0.999),
            weight_decay=0.0001,
        )
        lr_scheduler = get_scheduler(
            optimizer,
            self.hparams.lr_policy,
            self.hparams.n_ep_decay,
            self.hparams.num_epochs,
            self.hparams.lr,
        )
        if lr_scheduler is not None:
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]

    # def configure_for_training(self, opts):
    #     print(opts.__dict__)
    #     self.save_hyperparameters(opts.__dict__, ignore = ["model_path", "ipython_dir"])
    #     self.criterion = getattr(nn, self.hparams.nn_loss)()

    def predict_step(self, batch, batch_idx):
        if not hasattr(self, "prediction_type"):
            self.prediction_type = "latents"

        if self.prediction_type == "latents":
            imgs, y = batch["image"], batch["label"]
            record_name = batch["record_name"]
            output = self.forward(imgs)

            return {"latents": output, "id": record_name, "gt": y}
        elif self.prediction_type == "class":
            imgs, y = batch["image"], batch["label"]
            record_name = batch["record_name"]
            y_hat = self(imgs, return_class=True)
            preds = self.get_class_prediction(y_hat)
            # print({"predicted_class":preds, "id":record_name, "gt": y})
            return {"predicted_class": preds, "id": record_name, "gt": y}

    def test_step(self, batch, batch_idx):
        imgs, y = batch["image"], batch["label"]
        record_name = batch["record_name"]
        y_hat = self(imgs, return_class=True)
        preds = self.get_class_prediction(y_hat)
        # print({"predicted_class":preds, "id":record_name, "gt": y})
        return {"predicted_class": preds, "id": record_name, "gt": y}

    def get_loss(self, y_hat, y):
        return self.criterion(y_hat, y.to(torch.int64))

    def get_class_prediction(self, y_hat):

        preds = torch.argmax(F.softmax(y_hat.detach(), dim=1), dim=1)

        return preds

    # to do: continue from here
    def training_step(self, batch, batch_nb):
        imgs, y = batch["image"], batch["label"]
        # x, y = batch["x"], batch["y"]

        y_hat = self(imgs, return_class=True)

        loss = self.get_loss(y_hat, y)
        preds = self.get_class_prediction(y_hat)
        self.log("train_loss", loss.item(), prog_bar=True)

        if y.shape[0] > 1:
            f1 = f1_score(
                preds=preds,
                target=y.to(torch.int),
                task="multiclass",
                num_classes=self.model.num_classes,
            )
            self.log("train_f1", f1.item(), prog_bar=True)

            acc = accuracy(
                preds,
                y.to(torch.int),
                task="multiclass",
                num_classes=self.model.num_classes,
            )
            self.log("train_acc", acc.item(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        imgs, y = batch["image"], batch["label"]

        y_hat = self(imgs, return_class=True)

        loss = self.get_loss(y_hat, y)
        preds = self.get_class_prediction(y_hat)

        self.log("val_loss", loss.item(), prog_bar=True)

        if y.shape[0] > 1:
            f1 = f1_score(
                preds=preds,
                target=y.to(torch.int),
                task="multiclass",
                num_classes=self.model.num_classes,
            )
            self.log("val_f1", f1.item(), prog_bar=True)

            acc = accuracy(
                preds,
                y.to(torch.int),
                task="multiclass",
                num_classes=self.model.num_classes,
            )
            self.log("val_acc", acc.item(), prog_bar=True)

        return loss

    # def extract_features(self, dataloader):
    #     # dataloader = dm.predict_dataloader()
    #     outputs = self.predict(dataloader)
    #     latents_size = outputs["latents"].shape[1]
    #     results_path = self.get_result_path(dm)
    #     filename = (
    #         f"{dm.name}_{dm.resolution}_{self.model_id}_feature_size-{latents_size}"
    #     )
    #     os.makedirs(results_path, exist_ok=True)

    #     np.save(results_path / f"{filename}.npy", outputs["latents"])

    #     with open(results_path / f"{filename}.pkl", "wb") as f:
    #         pkl.dump(outputs, f)
    # @property
    # def get_result_path(self):
    #     result_path = RESULTS_DIR / self.model_id
    #     return result_path
