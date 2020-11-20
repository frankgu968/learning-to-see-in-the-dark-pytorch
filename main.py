import torch
from model.model import UNet
from dataset import LTSIDDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import transforms as trf
import numpy as np
import torch.optim as optim
import torch.nn as nn
import os.path
from pathlib import Path
from matplotlib import pyplot as plt

np.random.seed(0) # Deterministic random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dir = './data/short/'
truth_dir = './data/long/'
checkpoint_dir = './checkpoint/'
preprocess_dir = './data/preprocess/'
checkpoint_path = checkpoint_dir + 'checkpoint.t7'
patch_size = 512
save_interval = 5   # epochs
batch_size = 24
initial_learning_rate = 1e-4
epochs = 2000
visualize = False

# Set up dataset and dataloader
print('Loading dataset...')
train_dataset = LTSIDDataset(input_dir, truth_dir, preprocess_dir=preprocess_dir,
                        collection='train',
                        transforms=transforms.Compose([
                                                        trf.RandomCrop(patch_size),
                                                        trf.ToTensor(),
                                                        trf.RandomHorizontalFlip(p=0.5),
                                                        trf.RandomVerticalFlip(p=0.5),
                                                        trf.RandomTranspose(p=0.5),
                                                      ]))

test_dataset = LTSIDDataset(input_dir, truth_dir, preprocess_dir=preprocess_dir,
                        collection='test',
                        transforms=transforms.Compose([
                                                        trf.RandomCrop(patch_size),
                                                        trf.ToTensor(),
                                                        trf.RandomHorizontalFlip(p=0.5),
                                                        trf.RandomVerticalFlip(p=0.5),
                                                        trf.RandomTranspose(p=0.5),
                                                      ]))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print('Dataset loaded!')

# Set up model
model = UNet().to(device)

# Set up loss function
loss_func = nn.L1Loss()

# Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

# Learning rate scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs/2, gamma=0.1)

# Load model (if applicable)
start_epoch = 0
if os.path.exists(checkpoint_path):
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  start_epoch = checkpoint['epoch']

# Make model checkpoint dir
Path(checkpoint_dir).mkdir(exist_ok=True)

# Visualization
axarr = {}
if visualize:
  _, axarr = plt.subplots(1, 2)
  plt.show(block=False)

# Training loop
print('Starting training loop...')
train_len = len(train_loader)
test_len = len(test_loader)
for epoch in range(start_epoch, epochs):
  print('Starting epoch: %d' % epoch)

  # Run training loop
  training_loss = 0.0
  for idx, batch in enumerate(train_loader):
    train, truth = batch['train'].to(device), batch['truth'].to(device)
    optimizer.zero_grad()
    outputs = model(train)
    loss = loss_func(outputs, truth)
    loss.backward()
    optimizer.step()

    training_loss = training_loss + loss

    # Visualize current progress
    if visualize and idx == 0:
        plt.cla()
        axarr[0].imshow(batch['truth'][0].transpose(0, 2))
        axarr[1].imshow(outputs.data[0].cpu().transpose(0, 2))
        plt.draw()
        plt.pause(0.1)

    print('Training batch {} / {}'.format(idx+1, train_len))
  training_loss = training_loss / train_len # Scale the training loss

  # Run test loop
  with torch.no_grad():
    test_loss = 0.0
    for idx, batch in enumerate(test_loader):
      input, truth = batch['train'].to(device), batch['truth'].to(device)
      outputs = model(input)
      loss = loss_func(outputs, truth)
      test_loss = test_loss + loss
      print('Testing batch {} / {}'.format(idx + 1, test_len))
    test_loss = test_loss / test_len # Scale the test loss

  print('Epoch: %5d | Training Loss: %.4f | Test Loss: %.4f' % (epoch, training_loss, test_loss))

  # Update optimizer parameter(s), if applicable
  scheduler.step()

  # Save model
  if epoch % save_interval == 0:
      state = {
          'epoch': epoch,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
      }
      torch.save(state, checkpoint_path)
      print('Saved state to ', checkpoint_path)

print('Training complete!')