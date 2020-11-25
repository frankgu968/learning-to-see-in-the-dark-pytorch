from flask import Flask, request, Response
import torch
import rawpy
from app import app
from app.dataset import pack_raw
from app.infer import inferTransform
from app.model.model import UNet
import numpy as np
from PIL import Image


checkpoint_path = './app/checkpoint/checkpoint.t7'
output_path = './app/output/test.png'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load model
model = UNet().to(device)
model.load_state_dict(torch.load(checkpoint_path)["state_dict"])

#set model to evaluate mode
model.eval()

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        image_path = data['file']
        ratio = int(data['ratio'])

        print("image path: %s, ratio %d" % (image_path, ratio))
        raw = rawpy.imread(image_path) 
        im = pack_raw(raw) * ratio
        im = inferTransform(im)

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

        return output_path