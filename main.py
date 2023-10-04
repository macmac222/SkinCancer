import streamlit as st
from PIL import Image
from torchmetrics.classification import BinaryConfusionMatrix
import torch
from torch import nn, optim, tensor
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import math
import copy

st.write("""
# TechLabs Project 05: Skin Cancer Classification

Using CNN to classify whether a skin mole is benign or malignant
""")

st.sidebar.write("# CNN models")
st.sidebar.write("""
## Result 1

- Used model: Pretrained VGG16
- Optimiser: SGD with scheduler
- Loss Function: Cross Entropy Loss
- Batch size: 16
- Learning rate: 0.001
- Number of epochs: 20
(Early stopping)
- Test accuracy: ~89%

## Result 2

- Used model: Pretrained AlexNet
- Optimiser: SGD with scheduler
- Loss Function: Cross Entropy Loss
- Batch size: 30
- Learning rate: 0.001
- Number of epochs: 25
(Early stopping)
- Test accuracy: ~86%

""")

uploaded = st.file_uploader(label="", type=".jpg", accept_multiple_files=False)
# for image in uploaded:
#    image1 = Image.open(image)
#    print(type(image1))


def predict1(image):
    # load pretrained vgg16 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg16 = models.vgg16_bn(weights=None)

    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, 2)]) # Add our layer with 2 outputs
    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

    # Load pretrained weights
    vgg16.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
    vgg16.to(device)

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )])

    # load the image, pre-process it, and make predictions
    img = transform(image).unsqueeze(0)
    img = Variable(img)
    vgg16.eval()
    out = vgg16(img)
    _, predicted = torch.max(out, 1)
    return predicted.item()


def predict2(image):
    # load pretrained vgg16 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=None)

    # Freeze training for all layers
    for param in alexnet.features.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = alexnet.classifier[6].in_features
    features = list(alexnet.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, 2)])  # Add our layer with 2 outputs
    alexnet.classifier = nn.Sequential(*features)  # Replace the model classifier

    #  pretrained weights
    alexnet.load_state_dict(torch.load("model_weights2.pth", map_location=torch.device('cpu')))
    alexnet.to(device)

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    # Load the image, pre-process it, and make predictions
    img = transform(image).unsqueeze(0)
    img = Variable(img)
    alexnet.eval()
    out = alexnet(img)
    _, predicted = torch.max(out, 1)
    return predicted.item()


if uploaded is not None:
    # display image that user uploaded
    loaded_image = Image.open(uploaded)
else:
    loaded_image = Image.open("default_malignant.jpg")

st.image(loaded_image, caption = 'Uploaded Image', use_column_width = True)
st.write("")
st.write("Making predictions ...")
label1 = predict1(loaded_image)
if label1 == 1:
    result1 = "MALIGNANT"
else:
    result1 = "BENIGN"
st.subheader("Result 1 (VGG16): ")
st.write(result1)

label2 = predict2(loaded_image)
if label2 == 1:
    result2 = "MALIGNANT"
else:
    result2 = "BENIGN"
st.subheader("Result 2 (AlexNet): ")
st.write(result2)
