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
import os

# AWS Session
session = boto3.Session(
    aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY'],
    region_name = os.environ['AWS_REGION']
)

checkpoint_path = 'checkpoint/checkpoint.t7'

# Setting Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
model = UNet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])

#set model to evaluate mode
model.eval()

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieving Data from POST request
        print("Request recieved")
        data = request.get_json()
        bucketName = data['Bucket']
        inputImage = data['input-image']
        outputImage = data['output-image']
        ratio = int(data['ratio'])
        
        # Downloading input image from S3
        print("Importing Image")
        s3 = session.resource('s3')
        object = s3.Object(bucketName, inputImage)
        
        image = io.BytesIO()
        object.download_fileobj(image)
        image.seek(0)

        # image = response['Body']
        print("Image Imported")

        # Read and Transform
        print("Read Image")
        image = np.asarray(Image.open(image))
              
        print("Filter Black")
        im = np.maximum(image - 0.0, 0) / (255.0 - 0.0)  # subtract the black level
        
        print("Cast to Float32")
        im = im.astype(np.float32)

        print("Multiply by Ratio")
        im *= ratio

        print("Transform Image")
        im = inferTransform(im)
        
        print("Load Tensor")
        tensor = torch.from_numpy(im).transpose(0, 2).unsqueeze(0)
        tensor = tensor.to(device)

        with torch.no_grad():    
            # Inference
            print("Start Inference")
            output = model(tensor)
            print("Inference Complete")

            # Post processing for RGB output
            output = output.to('cpu').numpy() * 255
            output = output.squeeze()
            output = np.transpose(output, (2, 1, 0)).astype('uint8')
            output = Image.fromarray(output).convert("RGB")
            #output.show()

            # Output buffer for upload to S3
            buffer = io.BytesIO()            
            output.save(buffer, "PNG")
            buffer.seek(0) # rewind pointer back to start

            print("Exporting Image")
            s3.Bucket(bucketName).put_object(
                Key=outputImage,
                Body=buffer,
                ContentType='image/png',
            )
            print("Image Exported")

        return "Upload to S3 Complete"
        