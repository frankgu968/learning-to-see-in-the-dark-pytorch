import torch
from torch.utils.data import DataLoader
from dataset import LTSIDDataset
import transforms as trf
import torchvision.transforms as transforms
from PIL import Image
from model.model import UNet
from pathlib import Path
import numpy as np
import torch.nn as nn
import yaml
from config import Config
import sys

if __name__ == "__main__":
  try:
    config_file = sys.argv[1] # Check for supplied config
  except:
    config_file = 'defaultConfig.yaml' # Use default config

  with open(config_file, "r") as ymlfile:
    yml_file = yaml.load(ymlfile)
  cfg = Config(yml_file)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dataset = LTSIDDataset(cfg.input_dir, cfg.truth_dir, preprocess=cfg.preprocess,
                           preprocess_dir=cfg.preprocess_dir, collection='test',
                          transforms=transforms.Compose([trf.ToTensor()]))
  dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

  # Load model
  model = UNet().to(device)
  model.eval() #Set model to evaluation mode

  # Set up loss function
  loss_func = nn.L1Loss()

  checkpoint = torch.load(cfg.checkpoint_path)
  model.load_state_dict(checkpoint['state_dict'])

  # Make model checkpoint dir
  Path(cfg.output_dir).mkdir(exist_ok=True)

  counter = 0
  with torch.no_grad():
    test_loss = 0.0
    test_len = len(dataloader)
    for idx, batch in enumerate(dataloader):
      test, truth = batch['train'].to(device), batch['truth'].to(device)
      outputs = model(test)
      loss = loss_func(outputs, truth)
      test_loss = test_loss + loss
      
      test_loss = test_loss / test_len
      print('Batch: %5d | Test Loss: %.4f' % (idx, test_loss))

      # Write to file
      for idx, output in enumerate(outputs.cpu().numpy() * 255):
        print('Processing image {}'.format(counter))
        output = np.transpose(output, (1, 2, 0)).astype('uint8')
        Image.fromarray(output).convert("RGB").save(cfg.output_dir+str(counter)+"_out.png")
        counter = counter + 1
    
    
