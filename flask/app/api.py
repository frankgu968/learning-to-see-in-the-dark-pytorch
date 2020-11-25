from flask import Flask, redirect, url_for, render_template, request, Response
from app import app
from app.model.model import UNet
import torch

checkpoint_path = './app/checkpoint/checkpoint.t7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load model
model = UNet().to(device)
model.load_state_dict(torch.load(checkpoint_path)["state_dict"])

#set model to evaluate mode
model.eval()

@app.route('/', methods=['GET'])
def hello():
    return 'Hello World!'