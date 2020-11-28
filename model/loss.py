import torch
from torchvision import models
from collections import namedtuple

# Set up the scaffolding of the pre-trained VGG16 module
class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        # Slice 1 -> layers 1-4 of VGG
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        # Slice 2 -> layers 4-9 of VGG
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        # Slice 3 -> layers 9-16 of VGG
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        # Slice 4 -> layers 16-23 of VGG
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.slice1(x)
        relu1_2 = out  # Snapshot output of relu1_2
        out = self.slice2(relu1_2)
        relu2_2 = out
        out = self.slice3(relu2_2)
        relu3_3 = out
        out = self.slice4(relu3_3)
        relu4_3 = out

        output_tuple = namedtuple("VGGOutputs", ['relu1_1', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = output_tuple(relu1_2, relu2_2, relu3_3, relu4_3)
        return out


def perceptual_loss(perceptual_model='vgg16', dist_func=torch.nn.MSELoss(), device='cpu'):
    model = {}

    # Instantiate different perceptual models (if applicable)
    if perceptual_model == 'vgg16':
        model = VGG16().to(device)

    def loss_func(model_output, ground_truth):
        model_embedding = model.forward(model_output)
        truth_embedding = model.forward(ground_truth)

        loss = dist_func(model_embedding.relu1_1, truth_embedding.relu1_1)
        loss = loss + dist_func(model_embedding.relu2_2, model_embedding.relu2_2)
        loss = loss + dist_func(model_embedding.relu3_3, model_embedding.relu3_3)
        loss = loss + dist_func(model_embedding.relu4_3, model_embedding.relu4_3)
        loss = loss / 4.0

        return loss

    return loss_func
