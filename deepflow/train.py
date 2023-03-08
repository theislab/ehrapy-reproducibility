import torch
import torchmetrics
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from model import DeepFlow
from dataset import Retinol
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", type=str, default="train", help="train or predict", required=True
)
args = parser.parse_args()

# define mode
mode = args.mode

# define saved model path
saved_path = "/home/icb/xinyue.zhang/ehrapy-reproducibility/deepflow/model/dual/epoch_0_iter_10.pth"

# define dataset path
annotations_file = (
    "/home/icb/xinyue.zhang/diabetic-retinopathy-detection/trainLabels.csv.zip"
)
img_dir = "/home/icb/xinyue.zhang/diabetic-retinopathy-detection/train"

# device specification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define hyperparameters
train_ratio = 0.98
num_epoch = 50
batch_size = 32
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0005
factor_epoch = 85
lr_factor = 0.1
patience = 5

# model output path
model_prefix = "model/dual"
os.makedirs(model_prefix, exist_ok=True)

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop((2048, 2048)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)


# Get dataset and dataloader
full_dataset = Retinol(annotations_file, img_dir=img_dir, transform=transform)
train_size = int(train_ratio * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


def train_ifc():

    model = DeepFlow()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=5).to(device)
    min_valid_loss = np.inf
    early_stop = 0
    for epoch in range(num_epoch):
        train_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader, 0)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # print training and validation loss for every 50 iterations
            if i % 50 == 49:
                print(
                    f"[Epoch {epoch + 1}, iter {i + 1:5d}] acc: {acc(outputs, labels)}, Training loss: {train_loss / 50:.3f}"
                )
                train_loss = 0.0
                valid_loss = 0.0
                model.eval()
                for _, data in enumerate(test_dataloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                print(
                    f"[Epoch {epoch + 1}, iter {i + 1:5d}] Validation loss: {valid_loss / 50:.3f}"
                )
                if min_valid_loss > valid_loss:
                    early_stop = 0
                    print(
                        f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})"
                    )
                    min_valid_loss = valid_loss
                else:
                    early_stop += 1
                    if early_stop == patience:
                        PATH = os.path.join(model_prefix, f"epoch_{epoch}_iter_{i}.pth")
                        print(f"Early Stop! Saving model to: {PATH}")
                        torch.save(model.state_dict(), PATH)



        PATH = os.path.join(model_prefix, f"epoch_{epoch}.pth")
        print(f"Saving model to: {PATH}")
        torch.save(model.state_dict(), PATH)

    print("Finished Training")


def predict(saved_path):
    model = DeepFlow()
    model.load_state_dict(torch.load(saved_path))
    model.to(device)

    model.eval()
    for i, data in enumerate(tqdm(test_dataloader, 0)):
        _, labels = data
        inputs, labels = data
        inputs = inputs.to(device)

        # Forward Pass
        outputs = model.first_part(inputs)
        outputs = outputs.to("cpu")
        outputs = pd.DataFrame(outputs.detach().numpy())
        labels = pd.DataFrame(labels.detach().numpy())
        print(labels)

        if i == 0:
            test_outputs = outputs
            test_labels = labels
        else:
            test_outputs = pd.concat((test_outputs, outputs), axis=0)
            test_labels = pd.concat((test_labels, labels), axis=0)

    test_labels.to_csv("labels.csv", sep=",")
    test_outputs.to_csv("features.csv", sep=",")
    print("Prediction completed, outputs dumped!")


if __name__ == "__main__":
    if mode == "train":
        print(f"Using {device} to train the model")
        train_ifc()
    else:
        print(f"Using {device} to get the features")
        predict(saved_path)
