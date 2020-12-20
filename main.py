import torch
from model.model import UNet
from model.loss import perceptual_loss
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
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
from torch.utils.tensorboard import SummaryWriter
from config import Config

def train(gpu_num, args, cfg):

  np.random.seed(0) # Deterministic random

  # Distributed Options
  dist_data_parallel = False
  rank_0_buffer = 2

  # Device configuration
  device = torch.device('cpu')

  if (cfg.num_nodes*cfg.gpus_per_node > 1):
    print("Running in distributed mode with " + str(cfg.gpus_per_node) + " gpus per node and " + str(cfg.num_nodes) + " nodes")
    dist_data_parallel = True
    rank = args.nr * cfg.gpus_per_node + gpu_num
    print("This node is rank " + str(rank))
    print("   Master at: " + str(os.environ.get('MASTER_ADDR')) + "  " + str(os.environ.get('MASTER_PORT')))
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank = rank
    )
    device = 0
    torch.cuda.set_device(device)
  elif(torch.cuda.is_available()):
    device = torch.device('cuda:0')


  # Set up model - check if running distributed
  #model = UNet()
  from model.wavelet_model import WaveletUNet
  model = WaveletUNet()
  if (dist_data_parallel):
    print("Distributing Data in Parallel")
    model.cuda(device)
  else:
    model.to(device)


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

  if (dist_data_parallel):
    print("Loading Distributed Data Set")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset, 
      num_replicas = args.world_size,
      rank = rank
    )
    validation_sampler = torch.utils.data.distributed.DistributedSampler(
      validation_dataset,
      num_replicas = args.world_size,
      rank = rank
    )

    # Ensure Rank 0 GPU has extra vmem to consolidate result
    batch_size = cfg.batch_size
    if (rank == 0):
      batch_size = batch_size - rank_0_buffer
       
    train_loader = DataLoader(
      train_dataset, 
      batch_size = batch_size,
      shuffle = False,
      num_workers = 0,
      pin_memory = True,
      sampler=train_sampler)
    validation_loader = DataLoader(
      validation_dataset, 
      batch_size = batch_size,
      shuffle = False,
      num_workers = 0,
      pin_memory = True,
      sampler=validation_sampler)
  print('Dataset loaded!')


  # Set up loss function
  loss_func = nn.L1Loss()
  # loss_func = perceptual_loss(perceptual_model='vgg16', dist_func=nn.MSELoss(), device=device) # Perceptual loss

  # Set up optimizer
  optimizer = optim.Adam(model.parameters(), lr=cfg.initial_learning_rate)

  # Run fp16 by default
  model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

  if(dist_data_parallel):
    model = DDP(model)
    
  # Learning rate scheduling
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.epochs/2, gamma=0.1)

  # Set up TensorBoard writer
  writer = SummaryWriter('runs/'+cfg.run_name, flush_secs=1)

  # Load model (if applicable) - by default load the latest 
  start_epoch = 0
  if os.path.exists(cfg.checkpoint_to_load):
    checkpoint = torch.load(cfg.checkpoint_to_load)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    start_epoch = checkpoint['epoch']

  # Make model checkpoint dir
  Path(cfg.checkpoint_dir).mkdir(exist_ok=True)

  # Track "Best" Checkpoint
  best_validation_loss = 100000.0
  golden_epoch = False

  # Training loop
  print('Starting training loop...')
  train_len = len(train_loader)
  validation_len = len(validation_loader)
  for epoch in range(start_epoch, cfg.epochs):
    print('Starting epoch: %d' % epoch)
    golden_epoch = False   
  
    # Loss function schedule (if applicable)
    # if epoch > 750:
    #   print("Using L1 Loss")
    #   loss_func = nn.L1Loss()

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
          print("Saving images to tensorboard...")
          writer.add_image('input', input[0], epoch)
          writer.add_image('output', outputs.data[0].cpu(), epoch)
          writer.add_image('ground truth', truth[0], epoch)
      validation_loss = validation_loss / validation_len # Scale the validation loss

      if(validation_loss < best_validation_loss and epoch % cfg.save_interval == 0):
        best_validation_loss = validation_loss
        golden_epoch = True
      

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

        # Keep the best epoch
        checkpoint_path = cfg.checkpoint_path
        if (golden_epoch):
            print('Best Epoch seen so far with validation loss of %.4f ' % (best_validation_loss))
            checkpoint_path = cfg.checkpoint_dir + 'best.t7'
            
        torch.save(state, checkpoint_path)
        print('Saved state to ', checkpoint_path)


  print('Training complete!')
  return



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('config_file', nargs='?', default='defaultConfig.yaml')
  parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
  parser.add_argument('-m', '--master', default='0.0.0.0', help='ip address of master')
  args = parser.parse_args()

  with open(args.config_file, "r") as ymlfile:
    yml_file = yaml.safe_load(ymlfile)
  cfg = Config(yml_file)

  args.world_size = cfg.gpus_per_node * cfg.num_nodes

  os.environ['MASTER_ADDR'] = args.master
  os.environ['MASTER_PORT'] = '8888'

  mp.spawn(train, nprocs=cfg.gpus_per_node, args=(args, cfg))

 

