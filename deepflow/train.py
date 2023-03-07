import torch
import torchmetrics
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import DeepFlow
from dataset import Retinol
from torch.utils.data import DataLoader
from tqdm import tqdm

# define mode
mode = "train"

# define saved model path
saved_path = None

# define dataset path
annotations_file = (
    "/home/icb/xinyue.zhang/diabetic-retinopathy-detection/trainLabels.csv.zip"
)
img_dir = "/home/icb/xinyue.zhang/diabetic-retinopathy-detection/train"

# device specification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define hyperparameters
train_ratio = 0.8
num_epoch = 1
batch_size = 64
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0005
factor_epoch = 85
lr_factor = 0.1

# model output path
model_prefix = "model/dual"


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
    min_valid_loss = np.inf
    for epoch in range(num_epoch):

        train_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader, 0)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            acc = torchmetrics.Accuracy(task="multiclass", num_classes=5).to(device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            end.record()
            torch.cuda.synchronize()
            print(f"one iter: {start.elapsed_time(end)/60000}")  # minutes

            if i % 2 == 1:  # print every 1000 mini-batches
                print(
                    f"[Epoch {epoch + 1}, iter {i + 1:5d}] acc: {acc(outputs, labels)}, loss: {train_loss / 50:.3f}"
                )
                
                train_loss = 0.0
                valid_loss = 0.0
                
                model.eval()  # Optional when not using Model Specific layer
                for i, data in enumerate(test_dataloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                
                    # Forward Pass
                    outputs = model(inputs)
                    # Find the Loss
                    loss = criterion(outputs, labels)
                    # Calculate Loss
                    valid_loss += loss.item()
                    if i == 0:
                        valid_outputs = outputs
                        valid_labels =labels
                    valid_outputs.append(outputs)
                    valid_labels.append(labels)
                valid_acc = acc(valid_outputs, valid_labels)
                print(f"validation accuracy: {valid_acc}")
                if min_valid_loss > valid_loss:
                    PATH = os.path.join(model_prefix, f"epoch_{epoch}_iter_{i}.pth")
                    print(
                        f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})"
                    )
                    print(f"Saving model to: {PATH}")
                    torch.save(model.state_dict(), PATH)
                    min_valid_loss = valid_loss

    print("Finished Training")


def predict(saved_path):
    model = DeepFlow()
    model.load_state_dict(torch.load(saved_path))
    model.eval()

    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward Pass
        outputs = model.first_part(inputs)

        print(outputs.shape)
        break
        # np.savetxt("features.csv", out[1], fmt='%.5f', delimiter=",")
        # print("Prediction completed, outputs dumped!")


if __name__ == "__main__":
    if mode == "train":
        print(f"Using {device} to train the model")
        train_ifc()
    else:
        print(f"Using {device} to get the features")
        predict()
