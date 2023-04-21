#!/usr/bin/env python
# coding: utf-8

# Training a Conv LSTM based Model for Image Segmentation

# In[1]:


# !conda env create -f environment.yaml


# In[2]:


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm


# In[3]:


# !gdown https://drive.google.com/uc?id=1I64DYabWlUU4D4ktAS8IMSrQxlaGJIKi


# In[4]:


# Following takes a really long time.
# !unzip Dataset_Student.zip &> /dev/null


# In[5]:


# Load image dataset
train_dataset = datasets.ImageFolder(root='Dataset_Student/train', 
                                     transform=transforms.ToTensor())
val_dataset = datasets.ImageFolder(root='Dataset_Student/val',
                                   transform=transforms.ToTensor())


# In[6]:


# np.unique([entry[0].shape for entry in train_dataset], return_counts=True)
# all the images have the same size


# In[7]:


mask = np.load('Dataset_Student/train/video_0/mask.npy')
print(mask.shape)
# mask also has the same shape as the images


# In[8]:


# Creating a custom dataset for frames
class SegmentationDataset(Dataset):
    def __init__(self, root_folder, transforms=None):
        self.root_folder = root_folder
        self.transforms = transforms
        
        # Get all the folders in the root folder
        self.video_folders = os.listdir(root_folder)
        self.video_folders.sort()
        self.video_folders = [os.path.join(root_folder, folder) for folder in self.video_folders]
        self.video_folders = [folder for folder in self.video_folders if os.path.isdir(folder)]
        
    def __len__(self):
        return len(self.video_folders) * 22
    
    def __getitem__(self, idx):
        folder = self.video_folders[idx//22]
        files = os.listdir(folder)
        files = [file for file in files if file.endswith('.png')]
        # sorted on numeric index
        files = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[1]))
        
        image = plt.imread(os.path.join(folder, files[idx%22]))
        # permute the dimensions to make it (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        # 22 masks for each video
        mask = np.load(os.path.join(folder, 'mask.npy'))[idx%22]
        # print(f"folder idx: {idx//22} image folder: {folder}, image file: {files[idx%22]}, mask idx: {idx%22}")
        
        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)
        
        return image, mask

transformations = transforms.Compose([
    torch.from_numpy,
    # transforms.Resize(3),
    ])

train_dataset = SegmentationDataset(root_folder='Dataset_Student/train', transforms=transformations)
val_dataset = SegmentationDataset(root_folder='Dataset_Student/val', transforms=transformations)


# In[9]:


train_dataset[0][0].shape, train_dataset[0][1].shape


# In[10]:


# for i in range(23):
#     print(train_dataset[i][0].shape, train_dataset[i][1].shape)


# In[11]:


len(train_dataset), len(val_dataset)


# In[12]:


# creating the dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)


# In[13]:


for batch in train_loader:
    print(batch[0].shape)
    print(batch[1].shape)
    break


# ## Encoder-Decoder Architecture

# In[25]:


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# In[26]:


class SegmentationModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, out_channels)
        
        self.relu = nn.ReLU()
    
    def forward(self, images):
        images = self.relu(self.conv1(images))
        images = self.relu(self.conv2(images))
        images = self.relu(self.conv3(images))
        images = F.max_pool2d(images, 2)
        images = images.permute(0, 2, 3, 1)
        images = self.relu(self.fc1(images))
        images = self.fc2(images)
        
        return images


# In[27]:


def validate(model, val_loader, criterion, device):
    accuracies, losses = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch
            images, masks = images.to(device), masks.long().to(device)
            outputs = model(images)
            outputs = transforms.functional.resize(outputs.permute(0, 3, 1, 2),
                                                   size=masks.shape[-2:], antialias=None)
            loss = criterion(outputs, masks)
            losses.append(loss.item())
            
            # calculate accuracy
            pred = torch.argmax(outputs, dim=1)
            correct = (pred == masks).sum().item()
            accuracies.append(correct / masks.numel())
    return np.mean(accuracies), np.mean(losses)


# In[28]:


# Train on the dataset
def train(model, train_loader, epochs, criterion, optimizer, 
          val_loader=None, scheduler=None, device='cpu', early_stopper=None):
    model.train()
    best_loss = np.inf
    for epoch in range(epochs):
        losses = []
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            images, masks = batch
            images, masks = images.to(device), masks.long().to(device)
            outputs = model(images)
            outputs = transforms.functional.resize(outputs.permute(0, 3, 1, 2),
                                                   size=masks.shape[-2:], antialias=None)
            loss = criterion(outputs, masks)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        stmt = f"Epoch {epoch+1}/{epochs} | Train Loss: {np.mean(losses):.4f}"
        if val_loader:
            val_acc, val_loss = validate(model, val_loader, criterion, device)
            stmt += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            
            if val_loss < best_loss:
                best_loss = val_loss
                print("Saving the best model")
                torch.save(model.state_dict(), f'best_model.pth')
                
            if early_stopper and early_stopper.early_stop(loss):
                print("Early stopping")
                break
        
        if scheduler:
            scheduler.step()
            stmt += f" | LR: {scheduler.get_last_lr()[0]:.6f}"
        
        print(stmt)


# In[29]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[18]:


import random
from torch.utils.data import Subset

# subset of train dataset and val dataset
# train_subset = Subset(train_dataset, range(0, 20))
# train_subset_loader = DataLoader(train_subset, batch_size=4, shuffle=True)

val_subset = Subset(val_dataset, random.sample(range(0, len(val_dataset)), 200))
val_subset_loader = DataLoader(val_subset, batch_size=4, shuffle=True)

model = SegmentationModel(3, 49).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
criterion = nn.CrossEntropyLoss()
early_stopper = EarlyStopper(patience=10, min_delta=0.001)
train(model, train_loader, epochs=100, 
      criterion=criterion, optimizer=optimizer, 
      val_loader=val_loader, scheduler=scheduler, 
      device=device, early_stopper=early_stopper)