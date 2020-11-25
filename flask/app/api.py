from flask import Flask, request, Response
import torch
import rawpy
from app import app
from app.dataset import pack_raw
from app.infer import inferTransform
from app.model.model import UNet
import numpy as np
from PIL import Image
import boto3
import io


checkpoint_path = './app/checkpoint/checkpoint.t7'

# Setting Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
model = UNet().to(device)
model.load_state_dict(torch.load(checkpoint_path)["state_dict"])

#set model to evaluate mode
model.eval()

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieving Data from POST request
        data = request.get_json()
        bucketName = data['Bucket']
        inputImage = data['input-image']
        outputImage = data['output-image']
        ratio = int(data['ratio'])
        
        # Downloading input image from S3
        s3 = boto3.resource('s3')
        obj = s3.Object(bucketName, inputImage)
        response = obj.get()
        image = response['Body']

        # Rawpy processing and transofrmation
        raw = rawpy.imread(image) 

        im = pack_raw(raw) * ratio
        im = inferTransform(im)

        tensor = torch.from_numpy(im).transpose(0, 2).unsqueeze(0)
        tensor = tensor.to(device)

        with torch.no_grad():    
            # Inference
            output = model(tensor)

            # Post processing for RGB output
            output = output.to('cpu').numpy() * 255
            output = output.squeeze()
            output = np.transpose(output, (2, 1, 0)).astype('uint8')
            output = Image.fromarray(output).convert("RGB")
            output.show()

            # Output buffer for upload to S3
            buffer = io.BytesIO()            
            output.save(buffer, "PNG")
            buffer.seek(0) # rewind pointer back to start
            s3.Bucket(bucketName).put_object(
                Key=outputImage,
                Body=buffer,
                ContentType='image/png',
            )

            print("Upload to S3 Complete")