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

# ## Encoder-Decoder Architecture

# In[18]:


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


# In[19]:


class SegmentationModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(in_channels, 64, 1)
        
        self.convtranspose1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.fc2 = nn.Linear(64, out_channels)
        
        self.relu = nn.ReLU()
    
    def forward(self, input):
        images = self.relu(self.conv1(input))
        images = self.relu(self.conv2(images))
        images = self.relu(self.conv3(images))
        images = F.max_pool2d(images, 2)
        images = self.relu(self.convtranspose1(images))
        images = images + self.relu(self.conv4(input))
        images = images.permute(0, 2, 3, 1)
        images = self.fc2(images)
        images = images.permute(0, 3, 1, 2)
        
        return images


# In[20]:


def validate_segmentation_model(model, val_loader, criterion, device):
    accuracies, losses = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch
            images, masks = images.to(device), masks.long().to(device)
            outputs = model(images)
            # outputs = transforms.functional.resize(outputs.permute(0, 3, 1, 2),
            #                                        size=masks.shape[-2:], antialias=None)
            loss = criterion(outputs, masks)
            losses.append(loss.item())
            
            # calculate accuracy
            pred = torch.argmax(outputs, dim=1)
            # only the pixels that are not background are considered
            pred = pred[masks != 0]
            masks = masks[masks != 0]
            correct = (pred == masks).sum().item()
            accuracies.append(correct / masks.numel())
    return np.mean(accuracies), np.mean(losses)


# In[21]:


# Train on the dataset
def train_segmentation_model(model, train_loader, epochs, criterion, optimizer, 
          val_loader=None, scheduler=None, device='cpu', early_stopper=None, save=True):
    model.train()
    best_loss = np.inf
    for epoch in range(epochs):
        losses = []
        for (images, masks) in tqdm(train_loader):
            optimizer.zero_grad()
            images, masks = images.to(device), masks.long().to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        stmt = f"Epoch {epoch+1}/{epochs} | Train Loss: {np.mean(losses):.4f}"
        if val_loader:
            val_acc, val_loss = validate_segmentation_model(model, val_loader, criterion, device)
            stmt += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            
            if val_loss < best_loss and save:
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


# In[22]:


device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device


# In[23]:


from torch.utils.data import Subset
import random



segmodel = SegmentationModel(3, 49)
segmodel.load_state_dict(torch.load('best_model.pth', device), strict=False)
segmodel.to(device)
optimizer = torch.optim.Adam(segmodel.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
criterion = nn.CrossEntropyLoss()
early_stopper = EarlyStopper(patience=100, min_delta=0.001)

# ## Resnet Encoder and Decoder


# In[32]:


from os import path
import sys
sys.path.append(path.abspath('./VPTR/'))

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights
from model import GDL, MSELoss, L1Loss, GANLoss
from utils import visualize_batch_clips, save_ckpt, load_ckpt, set_seed, AverageMeters, init_loss_dict, write_summary, resume_training


# In[33]:


encH, encW, encC = 8, 8, 528
img_channels = 1 #3 channels for BAIR datset
epochs = 50
N = 32
AE_lr = 2e-4
lam_gan = 0.01


# In[34]:


from pathlib import Path

input_channels, output_channels = 3, 49
VPTR_Enc = VPTREnc(input_channels, feat_dim = encC, n_downsampling = 3).to(device)
VPTR_Dec = VPTRDec(output_channels, feat_dim = encC, n_downsampling = 3, out_layer = 'Sigmoid').to(device) #Sigmoid for MNIST, Tanh for KTH and BAIR
# VPTR_Disc = VPTRDisc(output_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(device)
# init_weights(VPTR_Disc)
init_weights(VPTR_Enc)
init_weights(VPTR_Dec)

optimizer = torch.optim.Adam(params = list(VPTR_Enc.parameters()) + list(VPTR_Dec.parameters()), 
                             lr=AE_lr, betas = (0.5, 0.999))
# optimizer_G = torch.optim.Adam(params = list(VPTR_Enc.parameters()) + list(VPTR_Dec.parameters()), lr=AE_lr, betas = (0.5, 0.999))
# optimizer_D = torch.optim.Adam(params = VPTR_Disc.parameters(), lr=AE_lr, betas = (0.5, 0.999))

criterion = nn.MSELoss()
loss_name_list = ['AE_MSE', 'AE_GDL', 'AE_total', 'Dtotal', 'Dfake', 'Dreal', 'AEgan']
gan_loss = GANLoss('vanilla', target_real_label=1.0, target_fake_label=0.0).to(device)
loss_dict = init_loss_dict(loss_name_list)
mse_loss = MSELoss()
gdl_loss = GDL(alpha = 1)
ckpt_save_dir = Path('/scratch/ss14412/FutureFramesDL/VPTR_ckpts/Segm_ResNetAE_MSEGDLgan_ckpt')


# In[35]:


def cal_lossD(VPTR_Disc, fake_imgs, real_imgs, lam_gan):
    pred_fake = VPTR_Disc(fake_imgs.detach().flatten(0, 1))
    loss_D_fake = gan_loss(pred_fake, False)
    # Real
    pred_real = VPTR_Disc(real_imgs.flatten(0,1))
    loss_D_real = gan_loss(pred_real, True)
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5 * lam_gan

    return loss_D, loss_D_fake, loss_D_real
    
def cal_lossG(VPTR_Disc, fake_imgs, real_imgs, lam_gan):
    pred_fake = VPTR_Disc(fake_imgs.flatten(0, 1))
    loss_G_gan = gan_loss(pred_fake, True)
    
    AE_MSE_loss = mse_loss(fake_imgs, real_imgs)
    AE_GDL_loss = gdl_loss(real_imgs, fake_imgs)
    #AE_L1_loss = l1_loss(fake_imgs, real_imgs)

    loss_G = lam_gan * loss_G_gan + AE_MSE_loss + AE_GDL_loss

    return loss_G, loss_G_gan, AE_MSE_loss, AE_GDL_loss


# In[36]:


def single_iter(VPTR_Enc, VPTR_Dec, criterion, optimizer, sample, device, train_flag = True):
    past_frames, masks = sample
    orig_masks = masks
    masks = F.one_hot(masks.long(), num_classes=49).permute(0, 3, 1, 2).float()
    past_frames, masks = past_frames.unsqueeze(1), masks.unsqueeze(1)
    past_frames = past_frames.to(device)
    masks = masks.to(device)
    orig_masks = orig_masks.to(device)
    
    if train_flag:
        VPTR_Enc = VPTR_Enc.train()
        VPTR_Enc.zero_grad()
        VPTR_Dec = VPTR_Dec.train()
        VPTR_Dec.zero_grad()
        
        rec_frames = VPTR_Dec(VPTR_Enc(past_frames))
        # print(rec_frames.shape, masks.shape)
        # print(np.unique(rec_frames.detach().cpu().numpy(), return_counts=True))
        loss = criterion(rec_frames, masks)
        loss.backward()
        optimizer.step()
    else:
        VPTR_Enc = VPTR_Enc.eval()
        VPTR_Dec = VPTR_Dec.eval()
        with torch.no_grad():
            rec_frames = VPTR_Dec(VPTR_Enc(past_frames))
            loss = criterion(rec_frames, masks)
            
            pred = torch.argmax(rec_frames.squeeze(1), dim=1)
            # only the pixels that are not background are considered
            pred = pred[orig_masks != 0]
            orig_masks = orig_masks[orig_masks != 0]
            correct = (pred == orig_masks).sum().item()
            accuracy = correct / orig_masks.numel()
        
    # iter_loss_dict = {'AEgan': loss_G_gan.item(), 'AE_MSE': AE_MSE_loss.item(), 'AE_GDL': AE_GDL_loss.item(), 'AE_total': loss_G.item(), 'Dtotal': loss_D.item(), 'Dfake':loss_D_fake.item(), 'Dreal':loss_D_real.item()}
    iter_loss_dict = {'AE_total': loss.item(), 'accuracy': accuracy}
    
    return iter_loss_dict


# In[37]:



resume_ckpt = ckpt_save_dir.joinpath('epoch_65.tar')
loss_dict, start_epoch = resume_training({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec}, 
                                         {}, resume_ckpt, loss_name_list)
VPTR_Enc_seg = VPTR_Enc
VPTR_Dec_seg = VPTR_Dec

# ## Video Frame Dataset

# In[43]:

import torch.distributed as dist
import torch

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):

    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0


from torchvision import transforms

class VideoFrameDataset(Dataset):
    def __init__(self, root_folder, transforms=None, labeled=True, model=None):
        self.root_folder = root_folder
        self.transforms = transforms
        self.labeled = labeled
        self.model = model
        
        # Get all the folders in the root folder
        self.video_folders = os.listdir(root_folder)
        self.video_folders.sort()
        self.video_folders = [os.path.join(root_folder, folder) for folder in self.video_folders]
        print(f"Length of video folders: {len(self.video_folders)}")
        self.video_folders = [folder for folder in self.video_folders if os.path.isdir(folder)]
        print(f"Length of dir: {len(self.video_folders)}")
        
    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        if self.labeled:
            # print(video_folder, frame_idx)
            masks = np.load(os.path.join(video_folder, 'mask.npy'))
            # print(masks.shape)
            frames = masks[:11]
            label = masks[11:]
        else:
            files = [frame for frame in os.listdir(video_folder) if frame.endswith('.png')]
            files = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[1]))
            images = [plt.imread(os.path.join(video_folder, frame)) for frame in files]
            frames = np.transpose(np.array(images[:11]), (0, 3, 1, 2))
            label = np.transpose(np.array(images[11:]), (0, 3, 1, 2))
            
        if self.transforms:
            frames = self.transforms(frames)
            label = self.transforms(label)
        
        return frames, label
    
transformations = transforms.Compose([
    torch.from_numpy,
    # transforms.Resize(40, antialias=None)
])

"""
labeled_dataset = VideoFrameDataset('/Dataset_Student/train', transforms=transformations)
trick_dataset = VideoFrameDataset('/Dataset_Student/train', transforms=transforms.Compose([
                                          torch.from_numpy,
                                          transforms.Normalize(mean=[0.5061, 0.5045, 0.5008], 
                                                               std=[0.0571, 0.0567, 0.0614])
                                          ]), 
                                      labeled=False, model=segmodel)
"""
unlabeled_dataset = VideoFrameDataset('/Dataset_Student/unlabeled', 
                                      transforms=transforms.Compose([
                                          torch.from_numpy,
                                          transforms.Normalize(mean=[0.5061, 0.5045, 0.5008], 
                                                               std=[0.0571, 0.0567, 0.0614])
                                          ]), 
                                      labeled=False, model=segmodel)
#val_dataset = VideoFrameDataset('/Dataset_Student/val', transforms=transformations)
val_trick_dataset = VideoFrameDataset('/Dataset_Student/val', transforms=transforms.Compose([
    torch.from_numpy,
    transforms.Normalize(mean=[0.5061, 0.5045, 0.5008],
        std=[0.0571, 0.0567, 0.0614])
    ]),
    labeled=False, model=segmodel)


# In[44]:

"""
print(labeled_dataset[0][0].shape, labeled_dataset[0][1].shape)
print(unlabeled_dataset[0][0].shape, unlabeled_dataset[0][1].shape)
print(val_dataset[0][0].shape, val_dataset[0][1].shape)
print(len(labeled_dataset), len(unlabeled_dataset), len(val_dataset))
"""


# #### Attention Module

# In[63]:


def cal_lossD(VPTR_Disc, fake_imgs, real_imgs, lam_gan):
    pred_fake = VPTR_Disc(fake_imgs.detach().flatten(0, 1))
    loss_D_fake = gan_loss(pred_fake, False)
    # Real
    pred_real = VPTR_Disc(real_imgs.flatten(0,1))
    loss_D_real = gan_loss(pred_real, True)
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5 * lam_gan

    return loss_D, loss_D_fake, loss_D_real
    
def cal_lossT(VPTR_Disc, fake_imgs, real_imgs, fake_feats, real_feats, lam_pc, lam_gan):
    T_MSE_loss = mse_loss(fake_imgs, real_imgs)
    T_GDL_loss = gdl_loss(real_imgs, fake_imgs)
    T_PC_loss = bpnce(F.normalize(real_feats, p=2.0, dim=2), F.normalize(fake_feats, p=2.0, dim=2))

    if VPTR_Disc is not None:
        assert lam_gan is not None, "Please input lam_gan"
        pred_fake = VPTR_Disc(fake_imgs.flatten(0, 1))
        loss_T_gan = gan_loss(pred_fake, True)
        loss_T = T_GDL_loss + T_MSE_loss + lam_pc * T_PC_loss + lam_gan * loss_T_gan
    else:
        loss_T_gan = torch.zeros(1)
        loss_T = T_GDL_loss + T_MSE_loss + lam_pc * T_PC_loss
    
    return loss_T, T_GDL_loss, T_MSE_loss, T_PC_loss, loss_T_gan


# In[64]:
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
def single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, optimizer_T, optimizer_D, sample, device, train_flag = True):
    past_frames, future_frames = sample
    past_frames = past_frames.to(device)
    future_frames = future_frames.to(device)
    
    with torch.no_grad():
        past_gt_feats = VPTR_Enc(past_frames)
        future_gt_feats = VPTR_Enc(future_frames)
        
        # Replacing real future images with future masks
        future_frames_mask = VPTR_Dec(VPTR_Enc(future_frames))
        
    if train_flag:
        VPTR_Transformer = VPTR_Transformer.train()
        VPTR_Transformer.zero_grad(set_to_none=True)
        VPTR_Dec.zero_grad(set_to_none=True)

        with autocast():
            pred_future_feats = VPTR_Transformer(past_gt_feats)
            pred_frames = VPTR_Dec(pred_future_feats)
        
            if optimizer_D is not None:
                assert lam_gan is not None, "Input lam_gan"
                #update discriminator
                VPTR_Disc = VPTR_Disc.train()
                for p in VPTR_Disc.parameters():
                    p.requires_grad_(True)
                VPTR_Disc.zero_grad(set_to_none=True)
                loss_D, loss_D_fake, loss_D_real = cal_lossD(VPTR_Disc, pred_frames, future_frames, lam_gan)
                loss_D.backward()
                optimizer_D.step()
        
                #update Transformer (generator)
                for p in VPTR_Disc.parameters():
                    p.requires_grad_(False)

            pred_future_feats = VPTR_Transformer.NCE_projector(pred_future_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            future_gt_feats = VPTR_Transformer.NCE_projector(future_gt_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            loss_T, T_GDL_loss, T_MSE_loss, T_PC_loss, loss_T_gan = cal_lossT(VPTR_Disc, pred_frames, future_frames_mask, 
                    pred_future_feats, future_gt_feats, lam_pc, lam_gan)
        scaler.scale(loss_T).backward()
        nn.utils.clip_grad_norm_(VPTR_Transformer.parameters(), max_norm=max_grad_norm, norm_type=2)
        scaler.step(optimizer_T)
        scaler.update()

    else:
        if optimizer_D is not None:
            VPTR_Disc = VPTR_Disc.eval()
        VPTR_Transformer = VPTR_Transformer.eval()
        with torch.no_grad():
            pred_future_feats = VPTR_Transformer(past_gt_feats)
            pred_frames = VPTR_Dec(pred_future_feats)
            if optimizer_D is not None:
                loss_D, loss_D_fake, loss_D_real = cal_lossD(VPTR_Disc, pred_frames, future_frames, lam_gan)

            pred_future_feats = VPTR_Transformer.NCE_projector(pred_future_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            future_gt_feats = VPTR_Transformer.NCE_projector(future_gt_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            loss_T, T_GDL_loss, T_MSE_loss, T_PC_loss, loss_T_gan = cal_lossT(VPTR_Disc, pred_frames, future_frames_mask, 
                                                                              pred_future_feats, future_gt_feats, 
                                                                              lam_pc, lam_gan)
    
    if optimizer_D is None:        
        loss_D, loss_D_fake, loss_D_real = torch.zeros(1), torch.zeros(1), torch.zeros(1)
    
    iter_loss_dict = {'T_total': loss_T.item(), 'T_MSE': T_MSE_loss.item(), 
                      'T_gan': loss_T_gan.item(), 'T_GDL': T_GDL_loss.item(),  
                      'T_bpc':T_PC_loss.item(), 'Dtotal': loss_D.item(), 'Dfake':loss_D_fake.item(), 
                      'Dreal':loss_D_real.item()}
    
    return iter_loss_dict

def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()

# In[65]:


from model import VPTRFormerNAR

ckpt_save_dir = Path('/scratch/ss14412/FutureFramesDL/VPTR_ckpts/VF_MSEGDLgan_ckpt')
num_past_frames = 11
num_future_frames = 11
encH, encW, encC = 20, 30, 528
img_channels = 3
epochs = 1000
N = 1
#AE_lr = 2e-4
Transformer_lr = 1e-4
max_grad_norm = 1.0 
TSLMA_flag = False
rpe = True
padding_type = 'zero'

lam_gan = None #0.001
lam_pc = 0.1
# device = torch.device('cuda:0')

show_example_epochs = 10
save_ckpt_epochs = 2

# VPTR_Enc = VPTREnc(img_channels, feat_dim = encC, n_downsampling = 3, padding_type = padding_type).to(device)
# VPTR_Dec = VPTRDec(img_channels, feat_dim = encC, n_downsampling = 3, out_layer = 'Tanh', padding_type = padding_type).to(device)
VPTR_Enc = VPTR_Enc.eval()
VPTR_Dec = VPTR_Dec.eval()

VPTR_Disc = None
#VPTR_Disc = VPTRDisc(img_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(device)
#VPTR_Disc = VPTR_Disc.eval()
#init_weights(VPTR_Disc)
# init_weights(VPTR_Enc)
# init_weights(VPTR_Dec)
print("Model Initialized")

VPTR_Transformer = VPTRFormerNAR(num_past_frames, num_future_frames, encH=encH, encW = encW, d_model=encC, 
                                nhead=8, num_encoder_layers=4, num_decoder_layers=8, dropout=0.1, 
                                window_size=4, Spatial_FFN_hidden_ratio=4, TSLMA_flag = TSLMA_flag, rpe = rpe, 
                                device=device)
#cuda_count = torch.cuda.device_count()
#if cuda_count > 1:
#    print("Let's use", cuda_count, "GPUs!")
#    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#    VPTR_Transformer = nn.DataParallel(VPTR_Transformer)
VPTR_Transformer = VPTR_Transformer.to(device)

optimizer_D = None
#optimizer_D = torch.optim.Adam(params = VPTR_Disc.parameters(), lr = Transformer_lr, betas = (0.5, 0.999))
optimizer_T = torch.optim.AdamW(params = VPTR_Transformer.parameters(), lr = Transformer_lr)

Transformer_parameters = sum(p.numel() for p in VPTR_Transformer.parameters() if p.requires_grad)
print(f"NAR Transformer num_parameters: {Transformer_parameters}")

# In[66]:


from model import GDL, MSELoss, L1Loss, GANLoss, BiPatchNCE

loss_name_list = ['T_MSE', 'T_GDL', 'T_gan', 'T_total', 'T_bpc', 'Dtotal', 'Dfake', 'Dreal']
#gan_loss = GANLoss('vanilla', target_real_label=1.0, target_fake_label=0.0).to(device)
bpnce = BiPatchNCE(N, num_future_frames, encH, encW, 1.0).to(device)
loss_dict = init_loss_dict(loss_name_list)
mse_loss = MSELoss()
gdl_loss = GDL(alpha = 1)

#load the trained autoencoder, we initialize the discriminator from scratch, for a balanced training
# loss_dict, start_epoch = resume_training({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec}, 
#                                          {}, resume_AE_ckpt, loss_name_list)
list_epochs = [f for f in os.listdir(ckpt_save_dir) if f.endswith('.tar')]
sorted_list_epochs = sorted(list_epochs, key=lambda x: int(x.split('_')[1].split('.')[0]))
resume_ckpt = None
if len(sorted_list_epochs) > 0:
    print(f"Loading checkpoint from epoch {sorted_list_epochs[-1]}")
    resume_ckpt = ckpt_save_dir.joinpath(sorted_list_epochs[-1])
if resume_ckpt is not None:
    loss_dict, start_epoch = resume_training({'VPTR_Transformer': VPTR_Transformer}, 
                                             {'optimizer_T':optimizer_T}, resume_ckpt, 
                                             loss_name_list)

cuda_count = torch.cuda.device_count()
if cuda_count > 1:
    print("Let's use", cuda_count, "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    VPTR_Transformer = nn.DataParallel(VPTR_Transformer)
#VPTR_Transformer = VPTR_Transformer.to(device)

"""
optimizer_T = torch.optim.AdamW(params = VPTR_Transformer.parameters(), lr = Transformer_lr)

Transformer_parameters = sum(p.numel() for p in VPTR_Transformer.parameters() if p.requires_grad)
print(f"NAR Transformer num_parameters: {Transformer_parameters}")
"""
# In[67]:

from torch.utils.data.distributed import DistributedSampler

#labeled_subset = Subset(labeled_dataset, range(1))
# unlabeled_subset = Subset(unlabeled_dataset, range(1))
#trick_subset = Subset(trick_dataset, range(1))
#val_trick_subset = Subset(val_trick_dataset, range(1))
# train_subset_segmentation_model = Subset(train_dataset_segmentation_model, range(12))
# val_subset_segmentation_model = Subset(val_dataset_segmentation_model, range(12))

# labeled_loader = DataLoader(labeled_subset, batch_size=1, shuffle=True)
# trick_loader = DataLoader(trick_subset, batch_size=1, shuffle=True)
# val_trick_loader = DataLoader(val_trick_subset, batch_size=1, shuffle=True)
# unlabeled_loader = DataLoader(unlabeled_subset, batch_size=1, shuffle=True)
# val_loader = DataLoader(labeled_subset, batch_size=1, shuffle=True)
# train_loader_segmentation_model = DataLoader(train_subset_segmentation_model, 
#                                              batch_size=12, shuffle=True)
# val_loader_segmentation_model = DataLoader(val_subset_segmentation_model, 
#                                            batch_size=12, shuffle=True)
# labeled_loader = DataLoader(labeled_dataset, batch_size=8, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
val_trick_loader = DataLoader(val_trick_dataset, batch_size=4, shuffle=True)


# In[68]:

import gc
from datetime import datetime

gc.collect()
torch.cuda.empty_cache()
for epoch in range(1, epochs+1):
    gc.collect()
    torch.cuda.empty_cache()
    epoch_st = datetime.now()

    #Train
    EpochAveMeter = AverageMeters(loss_name_list)
    for idx, sample in enumerate(tqdm(unlabeled_loader), 0):
        iter_loss_dict = single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, 
                                     optimizer_T, optimizer_D, sample, device, train_flag = True)
        EpochAveMeter.iter_update(iter_loss_dict)
        
    loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag = True)
    # write_summary(summary_writer, loss_dict, train_flag = True)

    # if epoch % show_example_epochs == 0 or epoch == 1:
    #     NAR_show_samples(VPTR_Enc, VPTR_Dec, VPTR_Transformer, sample, ckpt_save_dir.joinpath(f'train_gifs_epoch{epoch}'))
            
    torch.cuda.empty_cache()
    #validation
    EpochAveMeter = AverageMeters(loss_name_list)
    for idx, sample in enumerate(tqdm(val_trick_loader), 0):
        iter_loss_dict = single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, 
                                     optimizer_T, optimizer_D, sample, device, train_flag = False)
        EpochAveMeter.iter_update(iter_loss_dict)
    loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag = False)
    # write_summary(summary_writer, loss_dict, train_flag = False)

    #if epoch % save_ckpt_epochs == 0:
    save_ckpt({'VPTR_Transformer': VPTR_Transformer}, {'optimizer_T': optimizer_T}, epoch, loss_dict, ckpt_save_dir)
    
    # if epoch % show_example_epochs == 0 or epoch == 1:
    #     for idx, sample in enumerate(test_loader, random.randint(0, len(test_loader) - 1)):
    #         NAR_show_samples(VPTR_Enc, VPTR_Dec, VPTR_Transformer, sample, 
    #                          ckpt_save_dir.joinpath(f'test_gifs_epoch{epoch}'))
    #         break
        
    epoch_time = datetime.now() - epoch_st

    print(f"epoch {epoch}, {EpochAveMeter.meters['T_total']}")
    time_calc = epoch_time.total_seconds()/3600. * (start_epoch + epochs - epoch)
    print(f"Estimated remaining training time: {time_calc} Hours")


