import os.path
import shutil
from torchsummary import summary
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pandas as pd

from sklearn.metrics import classification_report
import os
from pathlib import Path

import numpy as np

from diabetic_retinopathy.constants import RESULTS_DIR, PhaseType, label_map
from diabetic_retinopathy.latent_extractors.datasets.dataloader import RatinaDataoduleTL

from diabetic_retinopathy.latent_extractors.predictors.utils import (
    Stage,
    FinishCallback,
    NNModel,
)
from diabetic_retinopathy.latent_extractors.predictors.models.LatentExtractor import (
    Models,
    get_latent_extractor,
    ExportModules,
)
from diabetic_retinopathy.constants import RESULTS_DIR

# fro, diabetic_retinopathy.latent_extractor.predictors.utils import Stage, NNModel, FinishCallback


class NNExtractor:
    def __init__(self, opts):

        self.model_name = opts.model_name
        self.model_path = Path(opts.model_path)
        self.opts = opts

    def get_num_classes(self):
        return len(list(label_map.keys()))

    def init_model(self, stage=Stage.fit):

        if torch.cuda.is_available():
            self.accelerator = "gpu"
        else:
            self.accelerator = "cpu"

        # if self.model_type == NNModel.SpatialSSL:
        # ToDo : define models here

        self.model = get_latent_extractor(
            model_name=Models[self.opts.model],
            num_classes=self.get_num_classes,
            opts=self.opts if Stage.fit else None,
        )

        # if stage == Stage.fit:
        # self.model.configure_for_training(self.opts)

    def get_transforms(self):
        test_transforms = self.model.transform
        special_HF_transform = self.model.module == "HF"
        if self.opts.augmentations:
            self.model._set_augmentations()
            train_transforms = self.model.train_transforms
        else:
            train_transforms = test_transforms
        return train_transforms, test_transforms, special_HF_transform

    def get_augmentations(self, augmentations_list):
        if augmentations_list == "standart":
            transforms.Compose([])
            augmentations = None
        return augmentations

    def get_dm(self, phase: PhaseType = PhaseType.train, stage: Stage = Stage.fit):

        train_transforms, test_transforms, special_HF_transform = self.get_transforms()
        print(train_transforms, test_transforms, special_HF_transform)

        dm = RatinaDataoduleTL(
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            special_HF_transform=special_HF_transform,
            batch_size=self.opts.batch_size,
            num_workers=self.opts.num_workers,
            seed=self.opts.seed,
            opts=self.opts,
        )

        dm.setup(stage=stage.value, phase=phase)

        return dm

    def train(self, resume: bool = False):

        self.init_model()

        self.train_dm = self.get_dm(stage=Stage.fit)

        pl.seed_everything(self.opts.seed)
        # Ensure that all operations are deterministic on GPU (if used) for reproducibility
        torch.backends.cudnn.determinstic = self.opts.deterministic
        # print(self.model_path / "best_loss")

        best_loss_checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.model_path / "best_loss",
            filename="bestValLoss-{epoch}-{step}-{val_loss:.02f}",
            save_top_k=1,
            save_on_train_epoch_end=True,
        )

        last_checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_path / "last",
            save_on_train_epoch_end=True,
            save_last=True,
            filename="checkpoint-{epoch}-{step}",
        )

        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=self.model_path, name="tensorboard", version="fixed_version"
        )
        lr_logger = LearningRateMonitor(logging_interval="step", log_momentum=False)

        trainer = pl.Trainer(
            accelerator=self.accelerator,
            # gpus = -1,
            max_epochs=self.opts.num_epochs,
            default_root_dir=self.model_path,
            callbacks=[
                best_loss_checkpoint_callback,
                last_checkpoint_callback,
                lr_logger,
                FinishCallback(self.model_path),
            ],
            logger=[tb_logger],
            enable_progress_bar=True,
            log_every_n_steps=1,
            limit_train_batches=self.opts.limit_train_batches,
            limit_val_batches=self.opts.limit_val_batches,
            profiler=self.opts.profiler,
            gradient_clip_val=self.opts.gradient_clip_val,
        )

        if not resume:
            if self.opts.find_lr:
                tuner = pl.tuner.Tuner(trainer)

                # 3. Tune learning rate
                lr_finder = tuner.lr_find(
                    self.model,
                    train_dataloaders=self.train_dm.train_dataloader(),
                    val_dataloaders=self.train_dm.val_dataloader(),
                )

                new_lr = lr_finder.suggestion()
                if new_lr is None:
                    print(
                        "new_lr was not found. Using default lr: "
                        + str(self.model.hparams.lr)
                    )
                else:
                    print("new_lr: " + str(new_lr))
                    self.model.hparams.lr = new_lr

                # self.model.save_hyperparameters()

            trainer.fit(self.model, datamodule=self.train_dm)
        else:
            self.ckpt_path = self.find_best_ckpt(
                self.model_path, which_checkpoint="last"
            )

            trainer.fit(self.model, datamodule=self.train_dm, ckpt_path=self.ckpt_path)

        self.trainer = trainer
        # return trainer

    def predict_latents(self, dataloader):

        # meant for predicting latents
        self.model.prediction_type = "latents"

        trainer = pl.Trainer(
            inference_mode=True, limit_predict_batches=self.opts.limit_predict_batches
        )
        lat = trainer.predict(
            model=self.model, dataloaders=dataloader, ckpt_path=self.ckpt_path
        )

        latents = torch.cat([el["latents"] for el in lat]).cpu().detach().numpy()

        gt = np.concatenate([el["gt"] for el in lat])
        pat_id = np.concatenate([el["id"] for el in lat])

        return latents, {"id": pat_id, "gt_class": gt}

    def predict_class(self, dataloader):
        # meant for predicting latents
        # print(self.opts.limit_predict_batches)
        self.model.prediction_type = "class"

        trainer = pl.Trainer(
            inference_mode=True,
            limit_predict_batches=self.opts.limit_predict_batches,
            limit_test_batches=self.opts.limit_predict_batches,
        )

        cl = trainer.predict(
            model=self.model, dataloaders=dataloader, ckpt_path=self.ckpt_path
        )

        class_labels = (
            torch.cat([el["predicted_class"] for el in cl]).cpu().detach().numpy()
        )

        gt = np.concatenate([el["gt"] for el in cl])
        pat_id = np.concatenate([el["id"] for el in cl])

        return {"predicted_class": class_labels, "id": pat_id, "gt_class": gt}

    def get_latents(self, phase=PhaseType.test):
        dataloader = self.get_dm(stage=Stage.test, phase=phase).test_dataloader()
        latents, info_dict = self.predict_latents(dataloader)

        feat_cols = [f"feature-{feat_idx}" for feat_idx in range(latents.shape[1])]
        latent_df = pd.DataFrame(latents, columns=feat_cols)
        info_df = pd.DataFrame.from_records(info_dict)

        res_by_record_df = pd.merge(
            info_df, latent_df, left_index=True, right_index=True
        )
        res_by_record_df["gt_class_decoded"] = res_by_record_df.apply(
            lambda x: label_map[int(x["gt_class"])], axis=1
        )

        self.save_latents_predictions(res_pd=res_by_record_df, phase=phase.value)

        return res_by_record_df

    def get_performance(self, phase=PhaseType.test, save_pred=False):
        # ToDO: fix
        dataloader = self.get_dm(stage=Stage.test, phase=phase).test_dataloader()
        res_dict = self.predict_class(dataloader)
        res_by_record_df = pd.DataFrame.from_records(res_dict)

        res_by_record_df["gt_class_decoded"] = res_by_record_df.apply(
            lambda x: label_map[int(x["gt_class"])], axis=1
        )

        res_by_record_df["predicted_class_decoded"] = res_by_record_df.apply(
            lambda x: label_map[int(x["predicted_class"])], axis=1
        )

        if save_pred:
            self.save_class_predictions(res_pd=res_by_record_df, phase=phase.value)

        return res_by_record_df

    def load_model(self, inference=True):
        # ToDo: define appropriate models here
        model_specs = Models[self.opts.model].value
        if model_specs["module"] != ExportModules.ResNetFC:
            model = get_latent_extractor(model_name=Models[self.opts.model])
            self.model_name = self.opts.model
            self.model_path = RESULTS_DIR / self.model_name

        else:
            self.ckpt_path = self.find_best_ckpt(
                os.path.join(self.model_path),
                which_checkpoint=self.opts.which_checkpoint,
            )

            model = get_latent_extractor(model_name=Models[self.opts.model])

            # state_dict = torch.load(ckpt_path, map_location=torch.device('cpu') )
            # model.load_state_dict(state_dict = state_dict["state_dict"])

        self.model = model
        if inference:
            self.model.eval()

    def save_latents_predictions(self, res_pd, phase):

        filepath = RESULTS_DIR / self.model_name / "latents"
        if self.opts.which_checkpoint is not None:
            filepath = filepath / self.opts.which_checkpoint

        os.makedirs(filepath, exist_ok=True)
        print(f"Saving under path: {filepath}")
        res_pd.to_csv(filepath / f"latents-{phase}.csv", index=False)

    def save_class_predictions(self, res_pd, phase):

        filepath = RESULTS_DIR / self.model_name / "class_predictions"
        if self.opts.which_checkpoint is not None:
            filepath = filepath / self.opts.which_checkpoint

        os.makedirs(filepath, exist_ok=True)
        print(f"Saving under path: {filepath}")
        res_pd.to_csv(filepath / f"{phase}_preds.csv", index=False)

    def find_best_ckpt(
        self, checkpoint_path, which_checkpoint, selection_metric="loss", ext=".ckpt"
    ):
        """
        Find the highest epoch in the Test Tube file structure.
        :param ckpt_folder: dir where the checpoints are being saved.
        :return: Integer of the highest epoch reached by the checkpoints.
        """

        if which_checkpoint == "best":
            checkpoint_path = os.path.join(checkpoint_path, f"best_{selection_metric}")
            avail_models = [
                str(Path(x).stem)
                for x in os.listdir(checkpoint_path)
                if ("epoch" in x and x.endswith(ext))
            ]

            avail_models_info = sorted(
                [
                    {
                        "fullname": x,
                        "epoch": int((x.split("-")[1].split("=")[-1])),
                        "step": int((x.split("-")[2].split("=")[-1])),
                        "loss": float((x.split("-")[3].split("=")[-1])),
                    }
                    for x in avail_models
                ],
                key=lambda d: (-d["loss"], d["epoch"]),
                reverse=True,
            )

        elif which_checkpoint == "last":
            checkpoint_path = os.path.join(checkpoint_path, "last")
            avail_models = [
                str(Path(x).stem)
                for x in os.listdir(checkpoint_path)
                if ("epoch" in x and x.endswith(ext))
            ]

            avail_models_info = sorted(
                [
                    {
                        "fullname": x,
                        "epoch": int((x.split("-")[1].split("=")[-1])),
                        "step": int((x.split("-")[2].split("=")[-1])),
                    }
                    for x in avail_models
                ],
                key=lambda d: (d["epoch"], d["step"]),
                reverse=True,
            )

        last_avail_model = avail_models_info[0]["fullname"]
        last_avail_model = os.path.join(checkpoint_path, f"{last_avail_model}{ext}")
        return last_avail_model
