import torch
import torch.nn as nn
import torch.nn.functional as F





class WaveletUNet(nn.MOdeul):
  def __init__(self):
    super().__init__()

    self.lrelu      = nn.LeakyReLU(0.2)
    self.maxpool    = nn.MaxPool2d(2)


