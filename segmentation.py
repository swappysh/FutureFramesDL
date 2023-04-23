#!/usr/bin/env python
# coding: utf-8

# Training a Conv LSTM based Model for Image Segmentation

# In[32]:


# !conda env create -f environment.yaml


# In[33]:


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


# In[34]:


# !gdown https://drive.google.com/uc?id=1I64DYabWlUU4D4ktAS8IMSrQxlaGJIKi
# !gdown https://drive.google.com/uc?id=1nhsaIqolamUPj3q34TeEBUdXg3oHZh3L


# In[35]:


# Following takes a really long time.
# !unzip Dataset_Student.zip &> /dev/null


# In[36]:


# Load image dataset
train_dataset = datasets.ImageFolder(root='Dataset_Student/train', 
                                     transform=transforms.ToTensor())
val_dataset = datasets.ImageFolder(root='Dataset_Student/val',
                                   transform=transforms.ToTensor())


# In[37]:


# np.unique([entry[0].shape for entry in train_dataset], return_counts=True)
# all the images have the same size


# In[38]:


mask = np.load('Dataset_Student/train/video_0/mask.npy')
print(mask.shape)
# mask also has the same shape as the images


# ## Segementation Dataset

# In[39]:


# # Creating a custom dataset for frames
# class SegmentationDataset(Dataset):
#     def __init__(self, root_folder, transforms=None):
#         self.root_folder = root_folder
#         self.transforms = transforms
        
#         # Get all the folders in the root folder
#         self.video_folders = os.listdir(root_folder)
#         self.video_folders.sort()
#         self.video_folders = [os.path.join(root_folder, folder) for folder in self.video_folders]
#         self.video_folders = [folder for folder in self.video_folders if os.path.isdir(folder)]
        
#     def __len__(self):
#         return len(self.video_folders) * 22
    
#     def __getitem__(self, idx):
#         folder = self.video_folders[idx//22]
#         files = os.listdir(folder)
#         files = [file for file in files if file.endswith('.png')]
#         # sorted on numeric index
#         files = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[1]))
        
#         image = plt.imread(os.path.join(folder, files[idx%22]))
#         # permute the dimensions to make it (C, H, W)
#         image = np.transpose(image, (2, 0, 1))
#         # 22 masks for each video
#         mask = np.load(os.path.join(folder, 'mask.npy'))[idx%22]
#         # print(f"folder idx: {idx//22} image folder: {folder}, image file: {files[idx%22]}, mask idx: {idx%22}")
        
#         if self.transforms:
#             image = self.transforms(image)
#             mask = self.transforms(mask)
        
#         return image, mask

# transformations = transforms.Compose([
#     torch.from_numpy,
#     # transforms.Resize(3),
#     ])

# train_dataset = SegmentationDataset(root_folder='Dataset_Student/train', transforms=transformations)
# val_dataset = SegmentationDataset(root_folder='Dataset_Student/val', transforms=transformations)


# In[40]:


# train_dataset[0][0].shape, train_dataset[0][1].shape


# In[41]:


# for i in range(23):
#     print(train_dataset[i][0].shape, train_dataset[i][1].shape)


# In[42]:


# len(train_dataset), len(val_dataset)


# In[43]:


# # creating the dataloaders
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)


# In[44]:


# for batch in train_loader:
#     print(batch[0].shape)
#     print(batch[1].shape)
#     break


# ## Encoder-Decoder Architecture

# In[45]:


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


# In[46]:


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


# In[47]:


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
            # only the pixels that are not background are considered
            pred = pred[masks != 0]
            masks = masks[masks != 0]
            correct = (pred == masks).sum().item()
            accuracies.append(correct / masks.numel())
    return np.mean(accuracies), np.mean(losses)


# In[48]:


# Train on the dataset
def train(model, train_loader, epochs, criterion, optimizer, 
          val_loader=None, scheduler=None, device='cpu', early_stopper=None, save=True):
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


# In[49]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[50]:


from torch.utils.data import Subset

# subset of train dataset and val dataset
train_subset = Subset(train_dataset, range(0, 4))
train_subset_loader = DataLoader(train_subset, batch_size=4, shuffle=True)

val_subset = Subset(val_dataset, range(0, 4))
val_subset_loader = DataLoader(val_subset, batch_size=4, shuffle=True)

segmodel = SegmentationModel(3, 49).to(device)
optimizer = torch.optim.Adam(segmodel.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
criterion = nn.CrossEntropyLoss()
early_stopper = EarlyStopper(patience=10, min_delta=0.001)
# train(segmodel, train_subset_loader, epochs=10, 
#       criterion=criterion, optimizer=optimizer, 
#       val_loader=val_subset_loader, scheduler=scheduler, 
#       device=device, early_stopper=early_stopper, save=False)


# In[51]:


# load pre-trained model
segmodel.load_state_dict(torch.load('best_model.pth', map_location=torch.device(device)))


# In[52]:


# # Run inference on one training image

# outputs = segmodel(train_dataset[10][0].unsqueeze(0).to(device))
# outputs = transforms.functional.resize(outputs.permute(0, 3, 1, 2),
#                                        size=train_dataset[0][1].shape[-2:], antialias=None)
# outputs = torch.argmax(outputs, dim=1).squeeze(0).detach().cpu()

# axis = plt.subplot(1, 2, 1)
# # axis.imshow(train_dataset[0][0].permute(1, 2, 0))
# # axis.imshow(train_dataset[0][1])
# axis.imshow(outputs)
# # print(outputs.shape, torch.argmax(outputs, dim=1).squeeze(0).detach().cpu().shape, train_dataset[0][1].shape)
# plt.show()


# In[53]:


# plt.imshow(train_dataset[10][1])


# ## Video Frame Dataset

# In[54]:


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
            masks = np.load(os.path.join(video_folder, 'mask.npy'))
            frames = masks[:11]
            label = masks[11:]
        else:
            images = [plt.imread(os.path.join(video_folder, frame)) for frame in os.listdir(video_folder)]
            images = np.stack(images, axis=0)
            images = np.moveaxis(images, -1, 1)
            images = torch.from_numpy(images)
            # frames = self.model(images.to(device)).cpu()
            # frames = transforms.functional.resize(frames.permute(0, 3, 1, 2),
            #                                       size=images.shape[-2:], antialias=None)
            # frames = torch.argmax(frames, dim=1)
            label = images[11:].numpy()
            frames = images[:11].numpy()
            
        if self.transforms:
            frames = self.transforms(frames)
            label = self.transforms(label)
        
        return frames, label
    
transformations = transforms.Compose([
    torch.from_numpy,
    transforms.Resize(40, antialias=None)
])

labeled_dataset = VideoFrameDataset('Dataset_Student/train', transforms=transformations)
unlabeled_dataset = VideoFrameDataset('Dataset_Student/unlabeled', 
                                      transforms=transforms.Compose([
                                          torch.from_numpy,
                                          transforms.Resize(80, antialias=None)
                                          ]), 
                                      labeled=False, model=segmodel)
val_dataset = VideoFrameDataset('Dataset_Student/val', transforms=transformations)


# In[55]:


print(labeled_dataset[0][0].shape, labeled_dataset[0][1].shape)
print(unlabeled_dataset[0][0].shape, unlabeled_dataset[0][1].shape)
print(val_dataset[0][0].shape, val_dataset[0][1].shape)
print(len(labeled_dataset), len(unlabeled_dataset), len(val_dataset))


# ## Convolutional LSTM implementation
# 
# Code from: https://github.com/ndrplz/ConvLSTM_pytorch

# In[56]:


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        # print(f"self.hidden_dim: {self.hidden_dim}, h_next: {h_next.shape}, c_next: {c_next.shape}")

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


# In[57]:


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
        elif len(hidden_state) != self.num_layers:
            raise NotImplementedError("Hidden state is not implemented for len(hidden_state) != self.num_layers")

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
            # print(f"layer_idx {layer_idx}, h {h.shape}, c {c.shape}")

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        # print("layer_output_list", layer_output_list[-1].shape)
        # print("last_state_list", last_state_list[-1][0].shape)
        # print(f"self.num_layers {self.num_layers}, self.hidden_dim {self.hidden_dim}")
        # print(f"self.cell_list {self.cell_list[-1].hidden_dim}")

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# In[58]:


class seq2seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, 
                 batch_first, bias, return_all_layers, segmodel=segmodel):
        super(seq2seq, self).__init__()
        self.segmodel = segmodel
        self.conv_lstm = ConvLSTM(input_dim, hidden_dim, kernel_size, 
                                  num_layers, batch_first, bias, return_all_layers)
        self.fc = nn.Linear(hidden_dim[-1], 49)
    
    def forward(self, x, masks, labeled=True):
        # x.shape: 4, 11, 49, 160, 240
        if not labeled:
            x = self.segmodel(torch.flatten(x, start_dim=0, end_dim=1))
            x = x.permute(0, 3, 1, 2)
            x = x.view(-1, 11, x.shape[1], x.shape[2], x.shape[3])
            
            masks = self.segmodel(torch.flatten(masks, start_dim=0, end_dim=1))
            masks = masks.permute(0, 3, 1, 2)
            masks = masks.view(-1, 11, masks.shape[1], masks.shape[2], masks.shape[3])

        for i in range(11):
            _, output = self.conv_lstm(x[:,-11:])
            # print(output[0][0].shape)
            output = self.fc(output[0][0].permute(0, 2, 3, 1))
            output = output.permute(0, 3, 1, 2).unsqueeze(1)
            x = torch.cat((x, output), dim=1)
        return x[:, -11:], masks


# ### Training

# In[59]:


# # Load checkpoint
# checkpoint = torch.load('best_model.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']


# In[60]:


def validate(model, val_loader, criteria, device=device):
    model.eval()
    losses, accuracies = [], []
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = F.one_hot(images.long(), num_classes=49).permute(0, 1, 4, 2, 3).to(device), masks.to(device)
            output, masks = model(images, masks)
            output = output.permute(0, 2, 1, 3, 4)
            loss = criteria(output, masks.long())
            pred = output.argmax(dim=1)
            pred = pred[masks != 0]
            masks = masks[masks != 0]
            prediction = (pred == masks).sum().item()
            accuracies.append(prediction/masks.numel())
            losses.append(loss.item())
    return np.mean(losses), np.mean(accuracies)


# In[61]:


def train(model, labeled_loader, unlabeled_loader, val_loader, optimizer, criteria, epochs, 
          device, scheduler, early_stopper=None, save=True):
    best_loss = np.inf
    model.train()
    for epoch in range(epochs):
        out_string = f"Epoch {epoch+1}/{epochs}"
        losses = []
        for batch_idx, (images, masks) in enumerate(tqdm(labeled_loader)):
            images = F.one_hot(images.long(), num_classes=49).permute(0, 1, 4, 2, 3)
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            output, masks = model(images, masks)
            loss = criteria(output.permute(0, 2, 1, 3, 4), masks.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        out_string += f", Labeled Loss: {np.mean(losses):.2f}"
        
        losses = []
        for batch_idx, (images, masks) in enumerate(tqdm(unlabeled_loader)):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            output, masks = model(images, masks, False)
            masks = masks.argmax(dim=2)
            loss = criteria(output.permute(0, 2, 1, 3, 4), masks.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        out_string += f", Unlabeled Loss: {np.mean(losses):.2f}"
        
        val_loss, mean_acc = validate(model, val_loader, criteria, device)
        if val_loss < best_loss and save:
                best_loss = val_loss
                print("Saving the best model")
                torch.save(model.state_dict(), f'complete_best_model.pth')
        out_string += f", Val Loss: {val_loss:.2f}, Mean Acc: {mean_acc:.4f}"
        out_string += f", LR: {scheduler.get_last_lr()[0]:.4f}"
                
        print(out_string)
        scheduler.step()
        
        if early_stopper is not None and early_stopper.early_stop(val_loss):
            print("Early stopping")
            break
            


# In[62]:


from torch.optim import lr_scheduler

model = seq2seq(input_dim=49,
             hidden_dim=[32, 32, 64],
             num_layers=3,
            #  hidden_dim=[64],
            #  num_layers=1,
             kernel_size=(3, 3),
             batch_first=True,
             bias=True,
             return_all_layers=False).to(device)

# hyperparameters
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
early_stopper = EarlyStopper(patience=100, min_delta=0.001)
epochs = 100


# In[63]:


BATCH = 20
# labeled_subset = Subset(labeled_dataset, range(0, 20))
# labeled_loader = DataLoader(labeled_subset, batch_size=BATCH, shuffle=True)

# unlabeled_subset = Subset(unlabeled_dataset, range(0, 40))
# unlabeled_loader = DataLoader(unlabeled_subset, batch_size=BATCH, shuffle=True)

labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH, shuffle=True)

val_subset = Subset(val_dataset, range(0, 200))
val_subset_loader = DataLoader(val_subset, batch_size=BATCH, shuffle=True)

# train_loader = DataLoader(labeled_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
# a = next(iter(train_loader))
# print(a[0].shape, a[1].shape)

train(model, labeled_loader, unlabeled_loader, val_subset_loader, optimizer, 
      criteria, epochs, device, scheduler)


# In[ ]:


# torch.save({
#             'epoch': epochs,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': 1187.166111328125,
#             }, 'best_model.pt')

# # load model from here --> 
# # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html


# In[ ]:




