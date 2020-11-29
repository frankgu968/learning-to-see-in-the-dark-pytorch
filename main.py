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
import sys
import yaml

class Config:
  def __init__(self, cfg):
    self.input_dir = cfg['input_dir']
    self.truth_dir = cfg['truth_dir']
    self.checkpoint_dir = cfg['checkpoint_dir']
    self.preprocess = bool(cfg['preprocess'])
    self.preprocess_dir = cfg['preprocess_dir']
    self.checkpoint_path = self.checkpoint_dir + 'checkpoint.t7'
    self.patch_size = int(cfg['patch_size'])
    self.save_interval = int(cfg['save_interval'])   # epochs
    self.batch_size = int(cfg['batch_size'])
    self.initial_learning_rate = float(cfg['initial_learning_rate'])
    self.epochs = int(cfg['epochs'])
    self.visualize = bool(cfg['visualize'])
    self.num_workers = int(cfg['num_workers'])


if __name__ == "__main__":
  try:
    config_file = sys.argv[1] # Check for supplied config
  except:
    config_file = 'defaultTrainConfig.yaml' # Use default config

  with open(config_file, "r") as ymlfile:
    yml_file = yaml.load(ymlfile)
  cfg = Config(yml_file)

  np.random.seed(0) # Deterministic random
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Set up dataset and dataloader
  print('Loading dataset...')
  train_dataset = LTSIDDataset(cfg.input_dir, cfg.truth_dir, preprocess=cfg.preprocess, 
                          preprocess_dir=cfg.preprocess_dir, collection='train',
                          transforms=transforms.Compose([
                                                          trf.RandomCrop(cfg.patch_size),
                                                          trf.ToTensor(),
                                                          trf.RandomHorizontalFlip(p=0.5),
                                                          trf.RandomVerticalFlip(p=0.5),
                                                          trf.RandomTranspose(p=0.5),
                                                        ]))

  validation_dataset = LTSIDDataset(cfg.input_dir, cfg.truth_dir, preprocess=cfg.preprocess, 
                          preprocess_dir=cfg.preprocess_dir, collection='validation',
                          transforms=transforms.Compose([
                                                          trf.RandomCrop(cfg.patch_size),
                                                          trf.ToTensor(),
                                                          trf.RandomHorizontalFlip(p=0.5),
                                                          trf.RandomVerticalFlip(p=0.5),
                                                          trf.RandomTranspose(p=0.5),
                                                        ]))
  train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
  validation_loader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
  print('Dataset loaded!')

  # Set up model
  model = UNet().to(device)

  # Set up loss function
  loss_func = nn.L1Loss()

  # Set up optimizer
  optimizer = optim.Adam(model.parameters(), lr=cfg.initial_learning_rate)

  # Learning rate scheduling
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.epochs/2, gamma=0.1)

  # Load model (if applicable)
  start_epoch = 0
  if os.path.exists(cfg.checkpoint_path):
    checkpoint = torch.load(cfg.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']

  # Make model checkpoint dir
  Path(cfg.checkpoint_dir).mkdir(exist_ok=True)

  # Visualization
  axarr = {}
  if cfg.visualize:
    _, axarr = plt.subplots(1, 3)
    plt.show(block=False)

  # Training loop
  print('Starting training loop...')
  train_len = len(train_loader)
  validation_len = len(validation_loader)
  for epoch in range(start_epoch, cfg.epochs):
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
      if cfg.visualize and idx == 0:
          plt.cla()
          axarr[0].imshow(batch['train'][0].transpose(0, 2))
          axarr[1].imshow(batch['truth'][0].transpose(0, 2))
          axarr[2].imshow(outputs.data[0].cpu().transpose(0, 2))
          plt.draw()
          plt.pause(0.1)

      print('Training batch {} / {}'.format(idx+1, train_len))
    training_loss = training_loss / train_len # Scale the training loss

    # Run validation loop
    with torch.no_grad():
      validation_loss = 0.0
      for idx, batch in enumerate(validation_loader):
        input, truth = batch['train'].to(device), batch['truth'].to(device)
        outputs = model(input)
        loss = loss_func(outputs, truth)
        validation_loss = validation_loss + loss
        print('Validating batch {} / {}'.format(idx + 1, validation_len))
      validation_loss = validation_loss / validation_len # Scale the validation loss

    print('Epoch: %5d | Training Loss: %.4f | Validation Loss: %.4f' % (epoch, training_loss, validation_loss))

    # Update optimizer parameter(s), if applicable
    scheduler.step()

    # Save model
    if epoch % cfg.save_interval == 0:
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, cfg.checkpoint_path)
        print('Saved state to ', cfg.checkpoint_path)

  print('Training complete!')
