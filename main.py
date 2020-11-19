import torch
from model.model import UNet
from dataset import LTSIDDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

# test_input = torch.rand(1, 4, 512, 512)
# model = UNet()
# output = model.forward(test_input)

# print(output.shape)
np.random.seed(0) # Deterministic random

train_dir = './data/short/'
truth_dir = './data/long/'
patch_size = 512
dataset = LTSIDDataset(train_dir, truth_dir, 
                        transforms=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.RandomCrop(patch_size),
                                                      ]))
dataloader = DataLoader(dataset, batch_size=15, shuffle=False)

dataloader_iterator = iter(dataloader)
data = next(dataloader_iterator)
print(data['train'].shape)
print(data['truth'].shape)

print(torch.all(torch.eq(data['truth'][1], data['truth'][0])))

