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
import sys
import yaml
from apex import amp
from torch.utils.tensorboard import SummaryWriter
from config import Config

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
  train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
  validation_loader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=True,)
  print('Dataset loaded!')

  # Set up model
  model = UNet().to(device)

  # Set up loss function
  loss_func = nn.L1Loss()

  # Set up optimizer
  optimizer = optim.Adam(model.parameters(), lr=cfg.initial_learning_rate)


  # Experiment with 16-bit precision
  amp.initialize(model, optimizer, opt_level='O2')

  # Learning rate scheduling
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.epochs/2, gamma=0.1)

  # Set up TensorBoard writer
  writer = SummaryWriter('runs/'+cfg.run_name, flush_secs=1)

  # Load model (if applicable)
  start_epoch = 0
  if os.path.exists(cfg.checkpoint_path):
    checkpoint = torch.load(cfg.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    start_epoch = checkpoint['epoch']

  # Make model checkpoint dir
  Path(cfg.checkpoint_dir).mkdir(exist_ok=True)

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

      # Amp handles mixed precision
      with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

      optimizer.step()

      training_loss = training_loss + loss

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

        if idx == 0:
          writer.add_image('input', input[0], epoch)
          writer.add_image('output', outputs.data[0].cpu(), epoch)
          writer.add_image('ground truth', truth[0], epoch)
      validation_loss = validation_loss / validation_len # Scale the validation loss

    print('Epoch: %5d | Training Loss: %.4f | Validation Loss: %.4f' % (epoch, training_loss, validation_loss))
    # Write to TensorBoard
    writer.add_scalar('training loss',
                      training_loss,
                      epoch)
    writer.add_scalar('validation loss',
                      validation_loss,
                      epoch)

    # Update optimizer parameter(s), if applicable
    scheduler.step()

    # Save model
    if epoch % cfg.save_interval == 0:
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict()
        }
        torch.save(state, cfg.checkpoint_path)
        print('Saved state to ', cfg.checkpoint_path)

  print('Training complete!')


