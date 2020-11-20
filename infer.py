import torch
from torch.utils.data import DataLoader
from dataset import LTSIDDataset
import transforms as trf
import torchvision.transforms as transforms
from PIL import Image
from model.model import UNet
from pathlib import Path
import numpy as np

batch_size = 2
raw_dir = './data/inference/input/'
truth_dir = './data/inference/reference/'
preprocess_dir = './data/inference/'
output_dir = './output/'
checkpoint_path = './checkpoint/checkpoint.t7'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LTSIDDataset(raw_dir, truth_dir, preprocess_dir=preprocess_dir,
                        collection='validation',
                        transforms=transforms.Compose([trf.ToTensor()]))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = UNet().to(device)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

# Make model checkpoint dir
Path(output_dir).mkdir(exist_ok=True)

counter = 0
with torch.no_grad():
  for idx, batch in enumerate(dataloader):
    images = batch['train'].to(device)
    outputs = model(images)

    # Write to file
    for idx, output in enumerate(outputs.cpu().numpy() * 255):
      print('Processing image {}'.format(idx))
      output = np.transpose(output, (2, 1, 0)).astype('uint8')
      ref = batch['truth'][idx]
      ref = np.transpose(ref, (2, 1, 0)).astype('uint8')
      Image.fromarray(output).convert("RGB").save(output_dir+str(counter)+"_out.png")
      Image.fromarray(ref).convert("RGB").save(output_dir + str(counter) + "_ref.png")
      counter = counter + 1
