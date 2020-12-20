import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.upsampling import Upsample

# Convolution Wrapper
def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):

  result = nn.Conv2d(in_channels,
                        out_channels,
                        kernel_size,
                        padding=(kernel_size//2)+dilation-1,
                        bias=bias,
                        dilation=dilation)
  return result


# Discret Wavelet Transform
class DWT(nn.Module):
  def __init__(self):
    super(DWT, self).__init__()
    self.requires_grad = False

  def dwt_init(self, x):
  
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

  def forward(self, x):
    return self.dwt_init(x)

# Inverse Wavelet Transform
class IWT(nn.Module):
  def __init__(self):
    super(IWT, self).__init__()
    self.requires_grad = False

  def iwt_init(self, x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
      in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    
    # Will need to fix this
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).half().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

  def forward(self, x):
    return self.iwt_init(x)

# The Model
class WaveletUNet(nn.Module):
  def __init__(self, conv=default_conv):
    super().__init__()

    # Model Parameters
    n_feats = 64
    nColor = 3
    kernel_size = 3
 
    # Try leaky ReLU
    self.activation = nn.LeakyReLU(0.2)
  
    # Signal Transforms - Wavelet Transform
    self.DWT = DWT()
    self.IWT = IWT()
  
    # Define the Blocks Used in the Model
    self.head = conv(nColor, n_feats, kernel_size)
  
    # Down the Unet
    self.down_l0_0 = conv(n_feats, n_feats, kernel_size, dilation=2)
    self.down_l0_1 = conv(n_feats, n_feats, kernel_size)
  
    self.down_l1_0 = conv(n_feats*4, n_feats*2, kernel_size)
    self.down_l1_1 = conv(n_feats*2, n_feats*2, kernel_size, dilation=2)
    self.down_l1_2 = conv(n_feats*2, n_feats*2, kernel_size)
  
    self.down_l2_0 = conv(n_feats*8, n_feats*4, kernel_size)
    self.down_l2_1 = conv(n_feats*4, n_feats*4, kernel_size, dilation=2)
    self.down_l2_2 = conv(n_feats*4, n_feats*4, kernel_size)
  
    # Bottom of the Unet
    self.pro_l3_0 = conv(n_feats*16, n_feats*8, kernel_size)
    self.pro_l3_1 = conv(n_feats*8, n_feats*8, kernel_size, dilation=2)
    self.pro_l3_2 = conv(n_feats*8, n_feats*8, kernel_size, dilation=3)
    self.pro_l3_3 = conv(n_feats*8, n_feats*8, kernel_size, dilation=3)
    self.pro_l3_4 = conv(n_feats*8, n_feats*8, kernel_size, dilation=2)
    self.pro_l3_5 = conv(n_feats*8, n_feats*16, kernel_size)
  
    # Up the Unet
    self.up_l2_0 = conv(n_feats*4, n_feats*4, kernel_size, dilation=2)
    self.up_l2_1 = conv(n_feats*4, n_feats*4, kernel_size)
    self.up_l2_2 = conv(n_feats*4, n_feats*8, kernel_size)
    
    self.up_l1_0 = conv(n_feats*2, n_feats*2, kernel_size, dilation=2)
    self.up_l1_1 = conv(n_feats*2, n_feats*2, kernel_size)
    self.up_l1_2 = conv(n_feats*2, n_feats*4, kernel_size)
  
    self.up_l0_0 = conv(n_feats, n_feats, kernel_size, dilation=2)
    self.up_l0_1 = conv(n_feats, n_feats, kernel_size)
  
    self.tail = conv(n_feats, nColor, kernel_size)

  # Connect the layers together
  def forward(self, x):

    x0 = self.activation(self.head(x))
    x0 = self.activation(self.down_l0_0(x0))
    x0 = self.activation(self.down_l0_1(x0))
  
    x1 = self.DWT(x0)
    x1 = self.activation(self.down_l1_0(x1))
    x1 = self.activation(self.down_l1_1(x1))
    x1 = self.activation(self.down_l1_2(x1))
  
    x2 = self.DWT(x1)
    x2 = self.activation(self.down_l2_0(x2))
    x2 = self.activation(self.down_l2_1(x2))
    x2 = self.activation(self.down_l2_2(x2))
  
    x3 = self.DWT(x2)
    x3 = self.activation(self.pro_l3_0(x3))
    x3 = self.activation(self.pro_l3_1(x3))
    x3 = self.activation(self.pro_l3_2(x3))
    x3 = self.activation(self.pro_l3_3(x3))
    x3 = self.activation(self.pro_l3_4(x3))
    x3 = self.activation(self.pro_l3_5(x3))
    x3 = self.IWT(x3) + x2
  
    x4 = self.activation(self.up_l2_0(x3))
    x4 = self.activation(self.up_l2_1(x4))
    x4 = self.activation(self.up_l2_2(x4))
    x4 = self.IWT(x4) + x1
  
    x5 = self.activation(self.up_l1_0(x4))
    x5 = self.activation(self.up_l1_1(x5))
    x5 = self.activation(self.up_l1_2(x5))
    x5 = self.IWT(x5) + x0
  
    x6 = self.activation(self.up_l0_0(x5))
    x6 = self.activation(self.up_l0_1(x6)) 
    x6 = self.tail(x6) + x

    # Loss Function expects larger output so upsample here
    resize = Upsample(scale_factor=2, mode='nearest')
    x7 = resize(x6)

    return x7

  

