import torch
from model.model import UNet
from dataset import LTSIDDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import transforms as trf
import numpy as np
import torch.optim as optim
import torch.nn as nn

np.random.seed(0) # Deterministic random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dir = './data/short/'
truth_dir = './data/long/'
patch_size = 512
batch_size = 4
initial_learning_rate = 1e-4
epochs = 4001

# Set up dataset and dataloader
print('Loading dataset...')
dataset = LTSIDDataset(train_dir, truth_dir, 
                        transforms=transforms.Compose([
                                                        trf.RandomCrop(patch_size),
                                                        trf.ToTensor(),
                                                        trf.RandomHorizontalFlip(p=0.5),
                                                        trf.RandomVerticalFlip(p=0.5),
                                                        trf.RandomTranspose(p=0.5),
                                                      ]))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print('Dataset loaded!')

# Set up model
model = UNet().to(device)

# Set up loss function
loss_func = nn.L1Loss()

# Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

# Training loop
print('Starting training loop...')
for epoch in range(epochs):
  print('Starting epoch: %d' % epoch)
  epoch_loss = 0.0
  for idx, data in enumerate(dataloader):
    train, truth = data['train'].to(device), data['truth'].to(device)
    optimizer.zero_grad()

    outputs = model(train)
    loss = loss_func(outputs, truth)
    loss.backward()
    optimizer.step()

    epoch_loss = epoch_loss + loss
    
  print('Epoch: %5d | Loss: %.3f' % (epoch, epoch_loss))

print('Training complete!')
# dataloader_iterator = iter(dataloader)
# data = next(dataloader_iterator)
# print(data['train'].shape)
# print(data['truth'].shape)

# print(torch.all(torch.eq(data['truth'][1], data['truth'][0])))

