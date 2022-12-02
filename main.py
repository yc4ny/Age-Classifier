import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
from train import train, adjust_learning_rate
from validate import validate 
from test import test

# Argparse for command line arguments
parser = argparse.ArgumentParser(description = "Age Classification")
parser.add_argument('--batch_size', type = int, default = 256, help = 'batch size')
parser.add_argument('--num_epochs', type = int, default = 20, help = 'number of epochs')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
parser.add_argument('--weight_decay', type = float, default = 1e-4, help = 'weight decay')
parser.add_argument('--num_workers', type = int, default = 16, help = 'number of workers')
parser.add_argument('--log_interval', type = int, default = 10, help = 'log interval')
parser.add_argument('--save_interval', type = int, default = 1, help = 'save interval')
parser.add_argument('--save_dir', type = str, default = 'checkpoints', help = 'save directory')
parser.add_argument('--log_dir', type = str, default = 'runs', help = 'log directory')
args = parser.parse_args()

# Parse Metadata
def parsing(meta_data):
    image_age_list = []
    for idx, row in meta_data.iterrows():
        image_path = row['image_path']
        age_class = row['age_class']
        image_age_list.append([image_path, age_class])
    return image_age_list

# Dataset Class
class Dataset(Dataset):
    def __init__(self, meta_data, image_directory, transform=None):
        self.meta_data = meta_data
        self.image_directory = image_directory
        self.transform = transform

        # process the meta data
        image_age_list = parsing(meta_data)

        self.image_age_list = image_age_list
        self.age_class_to_label = {
            "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7
        }

    def __len__(self):
        return len(self.meta_data)
               
    def __getitem__(self, idx):
        image_path, age_class = self.image_age_list[idx]
        img = Image.open(os.path.join(self.image_directory, image_path))
        label = self.age_class_to_label[age_class]

        if self.transform:
            img = self.transform(img)
        
        return img, label

# Dataset Labels to Age
label_to_age = {
    0: "0-6 years old",
    1: "7-12 years old",
    2: "13-19 years old",
    3: "20-30 years old",
    4: "31-45 years old",
    5: "46-55 years old",
    6: "56-66 years old",
    7: "67-80 years old"
}

# Data Files
train_meta_data_path = "./custom_korean_family_dataset_resolution_128/custom_train_dataset.csv"
train_meta_data = pd.read_csv(train_meta_data_path)
train_image_directory = "./custom_korean_family_dataset_resolution_128/train_images"

val_meta_data_path = "./custom_korean_family_dataset_resolution_128/custom_val_dataset.csv"
val_meta_data = pd.read_csv(val_meta_data_path)
val_image_directory = "./custom_korean_family_dataset_resolution_128/val_images"

test_meta_data_path = "./custom_korean_family_dataset_resolution_128/custom_test_dataset.csv"
test_meta_data = pd.read_csv(test_meta_data_path)
test_image_directory = "./custom_korean_family_dataset_resolution_128/test_images"

# Image Transformations 
train_transform = transforms.Compose([
    transforms.Resize(128), # Resize the image to 128x128
    transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
    transforms.ToTensor(),# Convert the image to a PyTorch Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize the image
])

val_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

test_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

# Transform Dataset Images
train_dataset = Dataset(train_meta_data, train_image_directory, train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

val_dataset = Dataset(val_meta_data, val_image_directory, val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

test_dataset = Dataset(test_meta_data, test_image_directory, test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Summary Writer for TensorBoard
writer = SummaryWriter()

# Learning Rate, Log Interval
learning_rate = args.learning_rate
log_step = args.log_interval

# Model
model = models.resnet50(pretrained=True) # pretrained ResNet50
# model = models.resnet152(pretrained=True) 
num_features = model.fc.in_features # get the number of features in the last layer
model.fc = nn.Linear(num_features, 8) # 8 classes
model = model.cuda() # GPU
criterion = nn.CrossEntropyLoss() # loss function
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

num_epochs = args.num_epochs # number of epochs
best_val_acc = 0 # best validation accuracy
best_epoch = 0 # best epoch

# Train
for epoch in range(num_epochs):
    adjust_learning_rate(optimizer,learning_rate, epoch)
    train_loss, train_acc = train(epoch, model, train_dataloader, optimizer, criterion, log_step)
    val_loss, val_acc = validate(epoch, model, val_dataloader, criterion, log_step)
    writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
    writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
# Save the best model
    if val_acc > best_val_acc:
        print("[Info] best validation accuracy!")
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), f'checkpoints/best_checkpoint.pth')

# Load the best model for testing
model = models.resnet50(pretrained=True)
# model = models.resnet152(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 8) # transfer learning
model = model.cuda()
model_path = 'checkpoints/best_checkpoint.pth'
model.load_state_dict(torch.load(model_path))

test_loss, test_acc = test(model, test_dataloader, criterion, log_step)
print("test loss:", test_loss)
print("test acc:", test_acc)
