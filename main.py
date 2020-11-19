import torch
from model.model import UNet
from dataset import LTSIDDataset, ToTensor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# test_input = torch.rand(1, 4, 512, 512)
# model = UNet()
# output = model.forward(test_input)

# print(output.shape)

train_dir = './data/short/'
truth_dir = './data/long/'
dataset = LTSIDDataset(train_dir, truth_dir, transforms=transforms.Compose([ToTensor()]))
dataloader = DataLoader(dataset, batch_size=15, shuffle=False)

dataloader_iterator = iter(dataloader)
data = next(dataloader_iterator)
print(data['train'].shape)
print(data['truth'].shape)

print(torch.all(torch.eq(data['truth'][0], data['truth'][11])))
print(torch.all(torch.eq(data['truth'][10], data['truth'][11])))
print(torch.all(torch.eq(data['truth'][11], data['truth'][12])))

