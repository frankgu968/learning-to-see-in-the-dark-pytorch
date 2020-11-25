import torch
from torch.utils.data import DataLoader
from dataset import pack_raw
import transforms as trf
import torchvision.transforms as transforms
from PIL import Image
from model.model import UNet
from pathlib import Path
import numpy as np
import rawpy
import math
import cv2
import time
import sys

checkpoint_path = './checkpoint/checkpoint.t7'
#image_path = './data/inference/input/20001_00_0.03s.ARW'
#ratio = 250

if __name__ == '__main__':
    image_path = sys.argv[1]
    ratio = int(sys.argv[2])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.eval()

    raw = rawpy.imread(image_path) 
    im = pack_raw(raw) * ratio

    if max(im.shape) > 1024:
        scale_factor = 1024 / max(im.shape)
        H = int(im.shape[0] * scale_factor)
        W = int(im.shape[1] * scale_factor)
        im = cv2.resize(im, (W,H), cv2.INTER_AREA)


    H = math.floor(im.shape[0]/16.0)*16
    W = math.floor(im.shape[1]/16.0)*16
    im = im[:H, :W, :]

    tensor = torch.from_numpy(im).transpose(0, 2).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():    
        output = model(tensor)
        output = output.to('cpu').numpy() * 255
        output = output.squeeze()
        output = np.transpose(output, (2, 1, 0)).astype('uint8')
        Image.fromarray(output).convert("RGB").show()
