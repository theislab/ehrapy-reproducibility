import torch
import torchmetrics
import os
import torchvision.transforms as transforms
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import pandas as pd
from model import DeepFlow, ConvNet
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
train_ratio = 0.90
num_epoch = 20
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

transform_1 = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop((2048, 2048)),
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ]
)


# Get dataset and dataloader
full_dataset = Retinol(annotations_file, img_dir=img_dir, transform=transform_1)
train_size = int(train_ratio * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


def train_ifc():

    model = ConvNet(num_classes=5).to(device)
    optimizer=Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function=nn.CrossEntropyLoss()
    print(f"train count: {len(train_dataloader)}")
    print(f"test count: {len(test_dataloader)}")
    #Model training and saving best model

    best_accuracy=0.0

    for epoch in range(num_epoch):
        
        #Evaluation and training on training dataset
        model.train()
        train_accuracy=0.0
        train_loss=0.0
        
        for i, (images,labels) in enumerate(tqdm(train_dataloader, 0)):
            inputs, labels = inputs.to(device), labels.to(device)
                
            optimizer.zero_grad()
            
            outputs=model(images)
            loss=loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
            
            
            train_loss+= loss.cpu().data*images.size(0)
            _,prediction=torch.max(outputs.data,1)
            
            train_accuracy+=int(torch.sum(prediction==labels.data))
            
        train_accuracy=train_accuracy/len(train_dataloader)
        train_loss=train_loss/len(train_dataloader)
        
        
        # Evaluation on testing dataset
        model.eval()
        
        test_accuracy=0.0
        for i, (images,labels) in enumerate(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
                
            outputs=model(images)
            _,prediction=torch.max(outputs.data,1)
            test_accuracy+=int(torch.sum(prediction==labels.data))
        
        test_accuracy=test_accuracy/len(test_dataloader)
        
        
        print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
        
        #Save the best model
        if test_accuracy>best_accuracy:
            torch.save(model.state_dict(),'best_checkpoint.model')
            best_accuracy=test_accuracy


def predict(saved_path):
    model = ConvNet()
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
