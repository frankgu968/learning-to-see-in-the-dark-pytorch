import torch
from model.model import UNet

test_input = torch.rand(1, 4, 512, 512)
model = UNet()
output = model.forward(test_input)

print(output.shape)