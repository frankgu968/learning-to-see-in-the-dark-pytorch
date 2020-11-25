import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
  def __init__(self):
    super().__init__()
    # Global leaky ReLU activation function
    self.lrelu      = nn.LeakyReLU(0.2)
    self.maxpool    = nn.MaxPool2d(2)

    self.conv1_0    = nn.Conv2d(3, 32, 3, padding=1)
    self.conv1_1    = nn.Conv2d(32, 32, 3, padding=1)
    self.conv2_0    = nn.Conv2d(32, 64, 3, padding=1)
    self.conv2_1    = nn.Conv2d(64, 64, 3, padding=1)
    self.conv3_0    = nn.Conv2d(64, 128, 3, padding=1)
    self.conv3_1    = nn.Conv2d(128, 128, 3, padding=1)
    self.conv4_0    = nn.Conv2d(128, 256, 3, padding=1)
    self.conv4_1    = nn.Conv2d(256, 256, 3, padding=1)
    self.conv5_0    = nn.Conv2d(256, 512, 3, padding=1)
    self.conv5_1    = nn.Conv2d(512, 512, 3, padding=1)

    self.upconv6_0  = nn.ConvTranspose2d(512,256,kernel_size=2, stride=2)
    self.conv6_1    = nn.Conv2d(512, 256, 3, padding=1)
    self.conv6_2    = nn.Conv2d(256, 256, 3, padding=1)

    self.upconv7_0  = nn.ConvTranspose2d(256,128,kernel_size=2, stride=2)
    self.conv7_1    = nn.Conv2d(256, 128, 3, padding=1)
    self.conv7_2    = nn.Conv2d(128, 128, 3, padding=1)

    self.upconv8_0  = nn.ConvTranspose2d(128,64,kernel_size=2, stride=2)
    self.conv8_1    = nn.Conv2d(128, 64, 3, padding=1)
    self.conv8_2    = nn.Conv2d(64, 64, 3, padding=1)

    self.upconv9_0  = nn.ConvTranspose2d(64,32,kernel_size=2, stride=2)
    self.conv9_1    = nn.Conv2d(64, 32, 3, padding=1)
    self.conv9_2    = nn.Conv2d(32, 32, 3, padding=1)

    self.conv10     = nn.Conv2d(32, 12, 3, padding=1)

  def upsample_and_concat(self, x1, x2, upconv_func):
    upconv = upconv_func(x1)
    out = torch.cat([x2, upconv], dim=1)
    
    return out

  def forward(self, x):
    conv1 = self.lrelu(self.conv1_0(x))
    conv1 = self.lrelu(self.conv1_1(conv1))
    max1  = nn.MaxPool2d(2)(conv1)

    conv2 = self.lrelu(self.conv2_0(max1))
    conv2 = self.lrelu(self.conv2_1(conv2))
    max2 = nn.MaxPool2d(2)(conv2)

    conv3 = self.lrelu(self.conv3_0(max2))
    conv3 = self.lrelu(self.conv3_1(conv3))
    max3 = nn.MaxPool2d(2)(conv3)

    conv4 = self.lrelu(self.conv4_0(max3))
    conv4 = self.lrelu(self.conv4_1(conv4))
    max4 = nn.MaxPool2d(2)(conv4)

    conv5 = self.lrelu(self.conv5_0(max4))
    conv5 = self.lrelu(self.conv5_1(conv5))

    upconv6 = self.upsample_and_concat(conv5, conv4, self.upconv6_0)
    conv6 = self.lrelu(self.conv6_1(upconv6))
    conv6 = self.lrelu(self.conv6_2(conv6))

    upconv7 = self.upsample_and_concat(conv6, conv3, self.upconv7_0)
    conv7 = self.lrelu(self.conv7_1(upconv7))
    conv7 = self.lrelu(self.conv7_2(conv7))

    upconv8 = self.upsample_and_concat(conv7, conv2, self.upconv8_0)
    conv8 = self.lrelu(self.conv8_1(upconv8))
    conv8 = self.lrelu(self.conv8_2(conv8))

    upconv9 = self.upsample_and_concat(conv8, conv1, self.upconv9_0)
    conv9 = self.lrelu(self.conv9_1(upconv9))
    conv9 = self.lrelu(self.conv9_2(conv9))

    conv10 = self.conv10(conv9)
    out = F.pixel_shuffle(conv10, 2)  # Depth to space
    return out
  
