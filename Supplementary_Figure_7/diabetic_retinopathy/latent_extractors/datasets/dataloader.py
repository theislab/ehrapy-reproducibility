import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from ehrapylat.constants import DATASPLIT_DIR, PROCESSED_DIR, PhaseType, RAWDATA_DIR
from torch.utils.data import WeightedRandomSampler
from ehrapylat.latent_extractors.predictors.utils import Stage
from torchvision import io
import pandas as pd


class RetinaDataset(Dataset):
    def __init__(
        self,
        transforms=None,
        special_HF_transform=False,
        augmentations=None,
        target_transform=None,
        phase=PhaseType.train,
    ):
        self.img_labels = pd.read_csv(str(DATASPLIT_DIR / f"{phase.value}_df.csv"))
        print(
            f"loaded labels from {str(DATASPLIT_DIR / f'{phase.value}_df.csv')}, shape: {self.img_labels.shape}"
        )
        self.img_dir = RAWDATA_DIR / "images"
        self.transforms = transforms
        self.special_HF_transform = special_HF_transform
        self.augmentations = augmentations
        self.target_transform = target_transform
        # self.load_transforms = load_transforms if load_transforms is not None else
        self.phase = phase

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
        image = self.read_img(self.img_dir / f"{row.image}.jpeg")
        label = row.level
        if self.transforms:
            image = (
                self.transforms(image, return_tensors="pt")["pixel_values"].squeeze(0)
                if self.special_HF_transform
                else self.transforms(image)
            )

        if self.target_transform:
            label = self.target_transform(label)

        return {"image": image, "label": label, "record_name": f"{row.image}"}

    def read_img(self, filepath):
        # print(filepath)
        image = io.read_image(str(filepath))
        if image.shape[-1] == 3:
            # print("permuting image")
            image = image.permute(0, 3, 1, 2)
        # print(image.type, image.shape)
        image = torch.div(image, 255.0)
        return image


class RatinaDataoduleTL(pl.LightningDataModule):
    def __init__(
        self,
        train_transforms=None,
        test_transforms=None,
        special_HF_transform=False,
        # augmentations = None,
        batch_size: int = 32,
        num_workers: int = 12,
        seed=5,
        opts=None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.dataset = dataset
        self.seed = seed
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.special_HF_transform = special_HF_transform
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)
        self.opts = opts

    def setup(self, stage=None, phase=None):
        if stage == "fit" or stage is None:
            self.dt_train = RetinaDataset(
                transforms=self.train_transforms,
                special_HF_transform=self.special_HF_transform,
                # augmentations = self.augmentations,
                phase=PhaseType.train,
            )

            self.dt_val = RetinaDataset(
                transforms=self.test_transforms,
                special_HF_transform=self.special_HF_transform,
                # augmentations = None,
                phase=PhaseType.val,
            )

        if stage == "test" or stage == "predict":
            self.dt_test = RetinaDataset(
                transforms=self.test_transforms,
                special_HF_transform=self.special_HF_transform,
                # augmentations = self.augmentations,
                phase=phase,
            )

    def train_dataloader(self):
        if self.opts.weighted_sampler:
            class_counts = self.dt_train.img_labels.level.value_counts()
            sample_weights = [
                1 / class_counts[i] for i in self.dt_train.img_labels.level.values
            ]
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=self.dt_train.img_labels.shape[0],
                replacement=True,
                generator=self.g,
            )
            return DataLoader(
                self.dt_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=train_sampler,
            )
        else:
            return DataLoader(
                self.dt_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                generator=self.g,
            )

    def val_dataloader(self):
        return DataLoader(
            self.dt_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=self.g,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dt_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            generator=self.g,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dt_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            generator=self.g,
        )
