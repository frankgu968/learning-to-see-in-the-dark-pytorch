import torch
from torch.utils.data import DataLoader
from app.dataset import pack_raw
import app.transforms as trf
import torchvision.transforms as transforms
from PIL import Image
from app.model.model import UNet
from pathlib import Path
import numpy as np
import rawpy
import math
import cv2
import time
import sys

checkpoint_path = './checkpoint/checkpoint.t7'

if __name__ == '__main__':
    try:
        image_path = sys.argv[1]
        output_path = sys.argv[2]
        ratio = int(sys.argv[3])
    except:
        print("Error in inference input, use command:\n $ python infer.py RAW_IMAGE_PATH OUTPUT_IMAGE EXPOSURE_RATIO")
        sys.exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    #set model to evaluate mode
    model.eval()

    # image import
    raw = rawpy.imread(image_path) 
    im = pack_raw(raw) * ratio
    im = inferTransform(im)

    # transpose and add dummy sample dimension
    tensor = torch.from_numpy(im).transpose(0, 2).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():    
        output = model(tensor)
        output = output.to('cpu').numpy() * 255
        output = output.squeeze()
        output = np.transpose(output, (2, 1, 0)).astype('uint8')
        output = Image.fromarray(output).convert("RGB")
        output.show()
        output.save(output_path)
        
def inferTransform(im):
    # scaling image down to a max dimension of 512, maintaining aspect ratio
    if max(im.shape) > 512:
        scale_factor = 512 / max(im.shape)
        H = int(im.shape[0] * scale_factor)
        W = int(im.shape[1] * scale_factor)
        im = cv2.resize(im, (W,H), cv2.INTER_AREA)


    # cropping image to nearest 16, to allow torch to compute
    H = math.floor(im.shape[0]/16.0)*16
    W = math.floor(im.shape[1]/16.0)*16
    im = im[:H, :W, :]

    return im