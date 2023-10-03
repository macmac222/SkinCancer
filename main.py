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

st.sidebar.subheader("CNN model parameters")
st.sidebar.text("""

Hyperparameters:
- Batch size: 16
- Learning rate: 0.001
- Number of epochs: 20 
- Number of classes: 2
- Number of channels: 3

Preprocessing images:
- Resize 256x256
- Center crop 224x224 
- Normalization

Optimizer:
- Custom CNN model: Adam
- VGG 16: SGD with scheduler

Loss Function:
- Cross Entropy Loss
""")

uploaded = st.file_uploader(label = "", type=".jpg", accept_multiple_files=False)
#for image in uploaded:
#    image1 = Image.open(image)
#    print(type(image1))

def predict(image):
    """Return top 5 predictions ranked by highest probability.

    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
    # load pretrained vgg16 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg16 = models.vgg16_bn(pretrained=False)

    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, 2)]) # Add our layer with 2 outputs
    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

    #Load pretrained weights
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
    img = Image.open(image)
    img = transform(img).unsqueeze(0)
    img = Variable(img)
    vgg16.eval()
    out = vgg16(img)
    _, predicted = torch.max(out, 1)
    return predicted.item()


if uploaded is not None:
    # display image that user uploaded
    image = Image.open(uploaded)
    st.image(image, caption = 'Uploaded Image', use_column_width = True)
    st.write("")
    st.write("Making prediction ...")
    label = predict(uploaded)
    if label == 1:
        result = "MALIGNANT"
    else:
        result = "BENIGN"
    st.subheader("Result: ")
    st.write(result)

##==================================LOAD CNN MODEL=====================================================


##======================================================================================================
